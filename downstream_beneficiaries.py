"""Calculate downstream beneficiaries."""
import argparse
import collections
import glob
import logging
import math
import multiprocessing
import os
import pathlib
import shutil
import sqlite3
import subprocess
import threading
import time
from inspect import signature
from functools import wraps
from multiprocessing import managers

from osgeo import gdal
from osgeo import osr
import ecoshard
import numpy
import pygeoprocessing
import pygeoprocessing.geoprocessing
import pygeoprocessing.routing
import taskgraph

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    filename='log.out',
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.WARN)
logging.getLogger('pygeoprocessing').setLevel(logging.WARN)


# Backport of https://github.com/python/cpython/pull/4819
# Improvements to the Manager / proxied shared values code
# broke handling of proxied objects without a custom proxy type,
# as the AutoProxy function was not updated.
#
# This code adds a wrapper to AutoProxy if it is missing the
# new argument.
orig_AutoProxy = managers.AutoProxy


@wraps(managers.AutoProxy)
def AutoProxy(*args, incref=True, manager_owned=False, **kwargs):
    # Create the autoproxy without the manager_owned flag, then
    # update the flag on the generated instance. If the manager_owned flag
    # is set, `incref` is disabled, so set it to False here for the same
    # result.
    autoproxy_incref = False if manager_owned else incref
    proxy = orig_AutoProxy(*args, incref=autoproxy_incref, **kwargs)
    proxy._owned_by_manager = manager_owned
    return proxy


def apply_manager_autopatch():
    if "manager_owned" in signature(managers.AutoProxy).parameters:
        return

    LOGGER.debug("Patching multiprocessing.managers.AutoProxy to add manager_owned")
    managers.AutoProxy = AutoProxy

    # re-register any types already registered to SyncManager without a custom
    # proxy type, as otherwise these would all be using the old unpatched
    # AutoProxy
    SyncManager = managers.SyncManager
    registry = managers.SyncManager._registry
    for typeid, (callable, exposed, method_to_typeid, proxytype) in \
            registry.items():
        if proxytype is not orig_AutoProxy:
            continue
        create_method = hasattr(managers.SyncManager, typeid)
        SyncManager.register(
            typeid,
            callable=callable,
            exposed=exposed,
            method_to_typeid=method_to_typeid,
            create_method=create_method,
        )


DEM_ZIP_URL = 'https://storage.googleapis.com/global-invest-sdr-data/global_dem_3s_md5_22d0c3809af491fa09d03002bdf09748.zip'

WATERSHED_VECTOR_ZIP_URL = 'https://storage.googleapis.com/ipbes-ndr-ecoshard-data/watersheds_globe_HydroSHEDS_15arcseconds_blake2b_14ac9c77d2076d51b0258fd94d9378d4.zip'

POPULATION_RASTER_URL_MAP = {
    '2000': 'https://storage.googleapis.com/ecoshard-root/population/lspop2000_md5_79a872e3480c998a4a8bfa28feee228c.tif',
    '2017': 'https://storage.googleapis.com/ecoshard-root/population/lspop2017_md5_2e8da6824e4d67f8ea321ba4b585a3a5.tif',
    }

HAB_MASK_URL = 'https://storage.googleapis.com/critical-natural-capital-ecoshards/habmasks/masked_all_nathab_esa2015_md5_50debbf5fba6dbdaabfccbc39a9b1670.tif'


WORKSPACE_DIR = 'workspace'
WATERSHED_WORKSPACE_DIR = os.path.join(WORKSPACE_DIR, 'watershed_workspace')
for dir_path in [WORKSPACE_DIR, WATERSHED_WORKSPACE_DIR]:
    os.makedirs(dir_path, exist_ok=True)
N_TO_STITCH = 100


def _warp_and_wgs84_area_scale(
        base_raster_path, model_raster_path, target_raster_path,
        interpolation_alg, clip_bb, watershed_vector_path, watershed_fid,
        working_dir):
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    model_raster_info = pygeoprocessing.get_raster_info(model_raster_path)
    clipped_base_path = '%s_clip%s' % os.path.splitext(target_raster_path)
    pygeoprocessing.warp_raster(
        base_raster_path, base_raster_info['pixel_size'],
        clipped_base_path, 'near',
        target_bb=clip_bb,
        vector_mask_options={
            'mask_vector_path': watershed_vector_path,
            'mask_vector_where_filter': f'"FID"={watershed_fid}'},
        working_dir=working_dir)

    lat_min, lat_max = clip_bb[1], clip_bb[3]
    _, n_rows = pygeoprocessing.get_raster_info(
        clipped_base_path)['raster_size']
    m2_area_per_lat = pygeoprocessing.geoprocessing._create_latitude_m2_area_column(
        lat_min, lat_max, n_rows)

    def _mult_op(base_array, base_nodata, scale, datatype):
        """Scale non-nodata by scale."""
        result = base_array.astype(datatype)
        if base_nodata is not None:
            valid_mask = ~numpy.isclose(base_array, base_nodata)
        else:
            valid_mask = numpy.ones(
                base_array.shape, dtype=bool)
        result[valid_mask] = result[valid_mask] * scale[valid_mask]
        return result

    scaled_raster_path = os.path.join(
        '%s_scaled%s' % os.path.splitext(clipped_base_path))
    base_pixel_area_m2 = model_raster_info['pixel_size'][0]**2
    # multiply the pixels in the resampled raster by the ratio of
    # the pixel area in the wgs84 units divided by the area of the
    # original pixel
    base_nodata = base_raster_info['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(clipped_base_path, 1), (base_nodata, 'raw'),
         base_pixel_area_m2/m2_area_per_lat,
         (numpy.float32, 'raw')], _mult_op,
        scaled_raster_path,
        gdal.GDT_Float32, base_nodata)

    pygeoprocessing.warp_raster(
        scaled_raster_path, model_raster_info['pixel_size'],
        target_raster_path, 'near',
        target_projection_wkt=model_raster_info['projection_wkt'],
        target_bb=model_raster_info['bounding_box'],
        working_dir=working_dir)
    os.remove(clipped_base_path)
    os.remove(scaled_raster_path)


def _create_outlet_raster(
        outlet_vector_path, base_raster_path, target_outlet_raster_path):
    """Create a raster that has 1s where outlet exists and 0 everywhere else.

    Args:
        outlet_vector_path (str): path to input vector that has 'i', 'j'
            fields indicating which pixels are outlets
        base_raster_path (str): path to base raster used to create
            outlet raster shape/projection.
        target_outlet_raster_path (str): created by this call, contains 0s
            except where pixels intersect with an outlet.

    Return:
        None.
    """
    pygeoprocessing.new_raster_from_base(
        base_raster_path, target_outlet_raster_path, gdal.GDT_Byte,
        [0])

    outlet_raster = gdal.OpenEx(
        target_outlet_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    outlet_band = outlet_raster.GetRasterBand(1)

    outlet_vector = gdal.OpenEx(outlet_vector_path, gdal.OF_VECTOR)
    outlet_layer = outlet_vector.GetLayer()

    one_array = numpy.ones((1, 1), dtype=numpy.int8)
    for outlet_feature in outlet_layer:
        outlet_band.WriteArray(
            one_array,
            outlet_feature.GetField('i'),
            outlet_feature.GetField('j'))
    outlet_band = None
    outlet_raster = None


def _mask_raster(base_raster_path, mask_raster_path, target_raster_path):
    """Mask base by mask."""
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]

    def _mask_op(base_array, mask_array):
        result = numpy.empty_like(base_array)
        result[:] = base_nodata
        nodata_mask = numpy.isclose(base_array, base_nodata)
        valid_mask = ~nodata_mask & (mask_array == 1)
        zero_mask = ~nodata_mask & (mask_array == 0)
        result[valid_mask] = base_array[valid_mask]
        result[zero_mask] = 0
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1),
         (mask_raster_path, 1)], _mask_op,
        target_raster_path, gdal.GDT_Float32, base_nodata)


def normalize(
        base_raster_path,
        weight_raster_path,
        target_raster_path):
    """Normalize base by weight."""
    base_nodata = pygeoprocessing.get_raster_info(base_raster_path)['nodata'][0]

    def _safe_div_op(base, weight):
        result = numpy.full(base.shape, base_nodata, dtype=numpy.float32)
        valid_mask = (base != base_nodata) & (weight > 0)
        result[valid_mask] = base[valid_mask]/weight[valid_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (weight_raster_path, 1)], _safe_div_op,
        target_raster_path, gdal.GDT_Float32, base_nodata)


def _sum_raster(raster_path):
    """Return the sum of the raster."""
    running_sum = 0
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    for _, data_array in pygeoprocessing.iterblocks((raster_path, 1)):
        if nodata is not None:
            valid_mask = ~numpy.isclose(data_array, nodata)
        else:
            valid_mask = slice(-1)
        running_sum += numpy.sum(data_array[valid_mask])
    return running_sum


def rescale_by_base(base_raster_path, new_raster_path, target_raster_path):
    """Target is new * base_max/new_max."""
    base_sum = _sum_raster(base_raster_path)
    new_sum = _sum_raster(new_raster_path)

    new_raster_info = pygeoprocessing.get_raster_info(new_raster_path)
    new_nodata = new_raster_info['nodata'][0]

    def _mult_op(new_array, scale):
        """Scale non-nodata by scale."""
        result = numpy.copy(new_array)
        if new_nodata is not None:
            result[~numpy.isfinite(result)] = new_nodata
            valid_mask = ~numpy.isclose(new_array, new_nodata)
        else:
            result[~numpy.isfinite(result)] = 0
            valid_mask = slice(-1)

        result[valid_mask] *= scale
        return result

    if new_sum == 0:
        scale = 1.0
    else:
        scale = base_sum / new_sum

    pygeoprocessing.raster_calculator(
        [(new_raster_path, 1), (scale, 'raw')], _mult_op,
        target_raster_path, new_raster_info['datatype'],
        new_raster_info['nodata'][0])


def process_watershed(
        job_id, watershed_vector_path, watershed_fid, dem_path, hab_path,
        pop_raster_path_list, target_beneficiaries_path_list,
        target_normalized_beneficiaries_path_list,
        target_hab_normalized_beneficiaries_path_list,
        target_stitch_work_queue_list):
    """Calculate downstream beneficiaries for this watershed.

    Args:
        job_id (str): unique ID identifying this job, can be used to
            create unique workspaces.
        watershed_vector_path (str): path to watershed vector
        watershed_fid (str): watershed FID to process
        dem_path (str): path to DEM raster
        hab_path (str): path to habitat mask raster
        pop_raster_path_list (list): list of population rasters to route
        target_beneficiaries_path_list (str): list of target downstream
            beneficiary rasters to create, parallel with
            `pop_raster_path_list`.
        target_normalized_beneficiaries_path_list (list): list of target
            normalized downstream beneficiary rasters, parallel with other
            lists.
        target_hab_normalized_beneficiaries_path_list (list): list of target
            hab normalized downstream beneficiary rasters, parallel with other
            lists.
        target_stitch_work_queue_list (list): list of work queue tuples to
            put done signals in when each beneficiary raster is done. The
            first element is for the standard target, the second for the
            normalized raster.

    Return:
        None.
    """
    working_dir = os.path.join(
        os.path.dirname(target_beneficiaries_path_list[0]))
    os.makedirs(working_dir, exist_ok=True)
    LOGGER.debug(f'create working directory for {job_id} at {working_dir}')

    task_graph = taskgraph.TaskGraph(working_dir, -1)

    watershed_info = pygeoprocessing.get_vector_info(watershed_vector_path)
    watershed_vector = gdal.OpenEx(watershed_vector_path, gdal.OF_VECTOR)
    watershed_layer = watershed_vector.GetLayer()
    watershed_feature = watershed_layer.GetFeature(watershed_fid)
    watershed_geom = watershed_feature.GetGeometryRef()
    watershed_centroid = watershed_geom.Centroid()
    utm_code = (
        math.floor((watershed_centroid.GetX() + 180)/6) % 60) + 1
    lat_code = 6 if watershed_centroid.GetY() > 0 else 7
    epsg_code = int('32%d%02d' % (lat_code, utm_code))
    epsg_sr = osr.SpatialReference()
    epsg_sr.ImportFromEPSG(epsg_code)

    watershed_envelope = watershed_geom.GetEnvelope()
    # swizzle the envelope order that by default is xmin/xmax/ymin/ymax
    lat_lng_watershed_bb = [watershed_envelope[i] for i in [0, 2, 1, 3]]
    target_watershed_bb = pygeoprocessing.transform_bounding_box(
        lat_lng_watershed_bb,
        watershed_info['projection_wkt'],
        epsg_sr.ExportToWkt())

    watershed_vector = None
    watershed_layer = None
    watershed_feature = None
    watershed_geom = None
    watershed_centroid = None
    watershed_envelope = None

    target_pixel_size = (300, -300)

    warped_dem_raster_path = os.path.join(working_dir, f'{job_id}_dem.tif')
    warped_habitat_raster_path = os.path.join(
        working_dir, f'{job_id}_hab.tif')
    align_task = task_graph.add_task(
        func=pygeoprocessing.align_and_resize_raster_stack,
        args=(
            [dem_path, hab_path],
            [warped_dem_raster_path, warped_habitat_raster_path],
            ['near', 'mode'], target_pixel_size, target_watershed_bb),
        kwargs={
            'target_projection_wkt': epsg_sr.ExportToWkt(),
            'vector_mask_options': {
                'mask_vector_path': watershed_vector_path,
                'mask_vector_where_filter': f'"FID"={watershed_fid}'},
            },
        target_path_list=[
            warped_dem_raster_path, warped_habitat_raster_path],
        task_name=(
            f'align and clip and warp dem/hab to {warped_dem_raster_path} '
            f'{warped_habitat_raster_path}'))

    filled_dem_raster_path = os.path.join(
        working_dir, f'{job_id}_filled_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (warped_dem_raster_path, 1), filled_dem_raster_path),
        kwargs={
            'working_dir': working_dir,
            'max_pixel_fill_count': 1000000},
        dependent_task_list=[align_task],
        target_path_list=[filled_dem_raster_path],
        task_name=f'fill dem pits to {filled_dem_raster_path}')

    flow_dir_mfd_raster_path = os.path.join(
        working_dir, f'{job_id}_flow_dir_mfd.tif')
    flow_dir_mfd_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_mfd,
        args=(
            (filled_dem_raster_path, 1), flow_dir_mfd_raster_path),
        kwargs={'working_dir': working_dir},
        dependent_task_list=[fill_pits_task],
        target_path_list=[flow_dir_mfd_raster_path],
        task_name=f'calc flow dir for {flow_dir_mfd_raster_path}')

    outlet_vector_path = os.path.join(
        working_dir, f'{job_id}_outlet_vector.gpkg')
    detect_outlets_task = task_graph.add_task(
        func=pygeoprocessing.routing.detect_outlets,
        args=((flow_dir_mfd_raster_path, 1), 'mfd', outlet_vector_path),
        dependent_task_list=[flow_dir_mfd_task],
        target_path_list=[outlet_vector_path],
        task_name=f'detect outlets {outlet_vector_path}')

    outlet_raster_path = os.path.join(
        working_dir, f'{job_id}_outlet_raster.tif')
    create_outlet_raster_task = task_graph.add_task(
        func=_create_outlet_raster,
        args=(
            outlet_vector_path, flow_dir_mfd_raster_path, outlet_raster_path),
        dependent_task_list=[detect_outlets_task],
        target_path_list=[outlet_raster_path],
        task_name=f'create outlet raster {outlet_raster_path}')

    flow_accum_mfd_raster_path = os.path.join(
        working_dir, f'{job_id}_flow_accum.tif')
    flow_accum_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=((flow_dir_mfd_raster_path, 1), flow_accum_mfd_raster_path),
        dependent_task_list=[flow_dir_mfd_task],
        target_path_list=[flow_accum_mfd_raster_path],
        task_name=f'calc upstream flow area for {flow_accum_mfd_raster_path}')

    hab_upstream_area_raster_path = os.path.join(
        working_dir, f'{job_id}_hab_upstream.tif')
    hab_upstream_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=((flow_dir_mfd_raster_path, 1), hab_upstream_area_raster_path),
        kwargs={'weight_raster_path_band': (warped_habitat_raster_path, 1)},
        dependent_task_list=[flow_dir_mfd_task],
        target_path_list=[hab_upstream_area_raster_path],
        task_name=(
            f'calc upstream hab area for {hab_upstream_area_raster_path}'))

    for (pop_raster_path, target_beneficiaries_path,
         target_normalized_beneficiaries_path,
         target_hab_normalized_beneficiaries_path,
         stitch_queue_tuple) in zip(
            pop_raster_path_list, target_beneficiaries_path_list,
            target_normalized_beneficiaries_path_list,
            target_hab_normalized_beneficiaries_path_list,
            target_stitch_work_queue_list):

        LOGGER.debug(f'processing {target_beneficiaries_path} and normalized')

        aligned_pop_raster_path = os.path.join(
            working_dir,
            f'''{job_id}_{os.path.basename(
                os.path.splitext(pop_raster_path)[0])}.tif''')

        pop_warp_task = task_graph.add_task(
            func=_warp_and_wgs84_area_scale,
            args=(
                pop_raster_path, warped_dem_raster_path,
                aligned_pop_raster_path,
                'near', lat_lng_watershed_bb,
                watershed_vector_path, watershed_fid, working_dir),
            dependent_task_list=[align_task],
            target_path_list=[aligned_pop_raster_path],
            task_name=f'align {aligned_pop_raster_path}')

        downstream_bene_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_mfd,
            args=(
                (flow_dir_mfd_raster_path, 1), (outlet_raster_path, 1),
                target_beneficiaries_path),
            kwargs={
                'weight_raster_path_band': (aligned_pop_raster_path, 1)},
            dependent_task_list=[
                pop_warp_task, create_outlet_raster_task, flow_dir_mfd_task],
            target_path_list=[target_beneficiaries_path],
            task_name=(
                'calc downstream beneficiaries for '
                f'{target_beneficiaries_path}'))

        # divide aligned_pop_raster_path by flow accum to get normalized then
        # route it downstream
        pop_normal_by_upstream_raster_path = '%s_norm%s' % os.path.splitext(
            aligned_pop_raster_path)

        normalize_by_dist_task = task_graph.add_task(
            func=normalize,
            args=(
                aligned_pop_raster_path, flow_accum_mfd_raster_path,
                pop_normal_by_upstream_raster_path),
            dependent_task_list=[flow_accum_task, align_task],
            target_path_list=[pop_normal_by_upstream_raster_path],
            task_name=(
                f'normalized beneficiaries for '
                f'{pop_normal_by_upstream_raster_path}'))

        prescaled_normalized_beneficiaries_path = (
            '%s_prescaled%s' % os.path.splitext(
                target_normalized_beneficiaries_path))
        downstream_norm_bene_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_mfd,
            args=(
                (flow_dir_mfd_raster_path, 1), (outlet_raster_path, 1),
                prescaled_normalized_beneficiaries_path),
            kwargs={
                'weight_raster_path_band': (pop_normal_by_upstream_raster_path, 1)},
            dependent_task_list=[
                pop_warp_task, create_outlet_raster_task, flow_dir_mfd_task,
                normalize_by_dist_task],
            target_path_list=[prescaled_normalized_beneficiaries_path],
            task_name=(
                'calc downstream normalized beneficiaries for '
                f'{prescaled_normalized_beneficiaries_path}'))

        task_graph.add_task(
            func=rescale_by_base,
            args=(
                aligned_pop_raster_path,
                prescaled_normalized_beneficiaries_path,
                target_normalized_beneficiaries_path),
            target_path_list=[target_normalized_beneficiaries_path],
            dependent_task_list=[downstream_norm_bene_task],
            task_name=f'rescale {target_normalized_beneficiaries_path}')

        # divide aligned_pop_raster_path by hab accum to get normalized by
        # hab then route it downstream
        pop_hab_normal_by_upstream_raster_path = (
            '%s_hab_norm%s' % os.path.splitext(
                aligned_pop_raster_path))

        normalize_by_dist_task = task_graph.add_task(
            func=normalize,
            args=(
                aligned_pop_raster_path, hab_upstream_area_raster_path,
                pop_hab_normal_by_upstream_raster_path),
            dependent_task_list=[hab_upstream_task, align_task],
            target_path_list=[pop_hab_normal_by_upstream_raster_path],
            task_name=(
                f'normalized beneficiaries for '
                f'{pop_hab_normal_by_upstream_raster_path}'))
        hab_pre_mask_normalized_beneficiaries_path = (
            '%s_pre_mask%s' % os.path.splitext(
                target_hab_normalized_beneficiaries_path))
        downstream_norm_hab_bene_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_mfd,
            args=(
                (flow_dir_mfd_raster_path, 1), (outlet_raster_path, 1),
                hab_pre_mask_normalized_beneficiaries_path),
            kwargs={
                'weight_raster_path_band': (
                    pop_hab_normal_by_upstream_raster_path, 1)},
            dependent_task_list=[
                pop_warp_task, create_outlet_raster_task, flow_dir_mfd_task,
                normalize_by_dist_task],
            target_path_list=[hab_pre_mask_normalized_beneficiaries_path],
            task_name=(
                'calc downstream normalized beneficiaries for '
                f'{hab_pre_mask_normalized_beneficiaries_path}'))
        # mask this result to the target
        prescaled_hab_normalized_beneficiaries_path = (
            '%s_prescaled%s' % os.path.splitext(
                target_hab_normalized_beneficiaries_path))
        mask_downstream_norm_bene_task = task_graph.add_task(
            func=_mask_raster,
            args=(
                hab_pre_mask_normalized_beneficiaries_path,
                warped_habitat_raster_path,
                prescaled_hab_normalized_beneficiaries_path),
            dependent_task_list=[
                downstream_norm_hab_bene_task,
                align_task],
            target_path_list=[prescaled_hab_normalized_beneficiaries_path],
            task_name=f'mask {prescaled_hab_normalized_beneficiaries_path}')

        task_graph.add_task(
            func=rescale_by_base,
            args=(
                aligned_pop_raster_path,
                prescaled_hab_normalized_beneficiaries_path,
                target_hab_normalized_beneficiaries_path),
            target_path_list=[target_hab_normalized_beneficiaries_path],
            dependent_task_list=[mask_downstream_norm_bene_task],
            task_name=f'rescale {target_hab_normalized_beneficiaries_path}')

        task_graph.join()
        stitch_queue_tuple[0].put(
            (target_beneficiaries_path, working_dir, job_id))
        stitch_queue_tuple[1].put(
            (target_normalized_beneficiaries_path, working_dir, job_id))
        stitch_queue_tuple[2].put(
            (target_hab_normalized_beneficiaries_path, working_dir, job_id))

    task_graph.close()
    task_graph.join()
    task_graph = None


def get_completed_job_id_set(db_path):
    """Return set of completed jobs, or initialize if not set."""
    if not os.path.exists(db_path):
        LOGGER.debug(f'dbpath: {db_path}')
        connection = sqlite3.connect(db_path)
        cursor = connection.execute(
            """
            CREATE TABLE completed_job_ids (
                job_id TEXT NOT NULL,
                PRIMARY KEY (job_id)
            );
            """)
        cursor.close()
        connection.commit()
        connection.close()
        cursor = None
        connection = None

    ro_uri = r'%s?mode=ro' % pathlib.Path(
        os.path.abspath(db_path)).as_uri()
    connection = sqlite3.connect(ro_uri, uri=True)
    cursor = connection.execute('''SELECT * FROM completed_job_ids''')
    result = set([_[0] for _ in cursor.fetchall()])
    cursor.close()
    connection.commit()
    connection.close()
    cursor = None
    connection = None
    return result


def job_complete_worker(
        completed_work_queue, work_db_path, clean_result, n_expected):
    """Update the database with completed work.

    Args:
        completed_work_queue (queue): queue with (working_dir, job_id)
            incoming from each stitched raster
        work_db_path (str): path to the work database
        clean_result (bool): if true, delete the working directory after
            ``n_expected`` results come through.
        n_expected (int): number of expected duplicate jobs to come through
            before marking complete.

    Return:
        ``None``
    """
    try:
        last_time = time.time()
        connection = sqlite3.connect(work_db_path)
        uncommited_count = 0
        processed_so_far = 0
        working_jobs = collections.defaultdict(int)
        global WATERSHEDS_TO_PROCESS_COUNT
        LOGGER.info(f'started job complete worker, initial watersheds {WATERSHEDS_TO_PROCESS_COUNT}')
        watersheds_per_sec_list = []
        while True:
            payload = completed_work_queue.get()
            if payload is None:
                LOGGER.info('got None in completed work, terminating')
                break
            working_dir, job_id = payload
            working_jobs[job_id] += 1
            if working_jobs[job_id] < n_expected:
                continue
            # we got n_expected, so mark complete
            del working_jobs[job_id]
            WATERSHEDS_TO_PROCESS_COUNT -= 1
            if clean_result:
                shutil.rmtree(working_dir, ignore_errors=True)
            cursor = connection.execute(
                f"""
                INSERT INTO completed_job_ids VALUES ("{job_id}")
                """)
            cursor.close()
            LOGGER.info(f'done with {job_id} {working_dir}')
            uncommited_count += 1
            if uncommited_count > N_TO_STITCH:
                connection.commit()
                current_time = time.time()
                watersheds_per_sec_list.append(
                    uncommited_count / (current_time - last_time))
                if len(watersheds_per_sec_list) > 30:
                    watersheds_per_sec_list.pop(0)
                watersheds_per_sec = numpy.mean(watersheds_per_sec_list)
                last_time = current_time
                processed_so_far += uncommited_count
                uncommited_count = 0
                remaining_time_s = (
                    WATERSHEDS_TO_PROCESS_COUNT / watersheds_per_sec)
                remaining_time_h = int(remaining_time_s // 3600)
                remaining_time_s -= remaining_time_h * 3600
                remaining_time_m = int(remaining_time_s // 60)
                remaining_time_s -= remaining_time_m * 60
                LOGGER.debug(f'watersheds_per_sec_list: {watersheds_per_sec_list}')
                LOGGER.info(
                    f'remaining watersheds to process: '
                    f'{WATERSHEDS_TO_PROCESS_COUNT} - '
                    f'processed so far {processed_so_far} - '
                    f'process/sec: {watersheds_per_sec:.1f} - '
                    f'time left: {remaining_time_h}:'
                    f'{remaining_time_m:02d}:{remaining_time_s:04.1f}')

        connection.commit()
        connection.close()
        cursor = None
        connection = None
    except Exception:
        LOGGER.exception('error on job complete worker')
        raise


def general_worker(work_queue):
    """Invoke func on args coming through work queue."""
    while True:
        payload = work_queue.get()
        if payload is None:
            work_queue.put(None)
            LOGGER.debug('got a none on general worker, quitting')
            break
        func, args = payload
        func(*args)


def stitch_worker(
        stitch_work_queue, target_stitch_raster_path,
        stitch_done_queue, clean_result):
    """Take jobs from stitch work queue and stitch into target."""
    stitch_buffer_list = []
    done_buffer = []
    n_buffered = 0
    while True:
        payload = stitch_work_queue.get()
        if payload is None:
            LOGGER.debug(f'stitch worker for {target_stitch_raster_path} got DONE signal')
            stitch_work_queue.put(None)
        else:
            raster_path, working_dir, job_id = payload
            done_buffer.append((working_dir, job_id))
            if not os.path.exists(raster_path):
                message = f'{raster_path} does not exist on disk when stitching into {target_stitch_raster_path} also working dir is {working_dir}'
                LOGGER.error(message)
                raise ValueError(message)
            stitch_buffer_list.append((raster_path, 1))
            n_buffered += 1
        if n_buffered > N_TO_STITCH or payload is None:
            LOGGER.info(
                f'about to stitch {n_buffered} into '
                f'{target_stitch_raster_path}')
            start_time = time.time()
            pygeoprocessing.stitch_rasters(
                stitch_buffer_list, ['near']*n_buffered,
                (target_stitch_raster_path, 1),
                area_weight_m2_to_wgs84=True,
                overlap_algorithm='etch')
            for working_dir, job_id in done_buffer:
                stitch_done_queue.put((working_dir, job_id))
            stitch_buffer_list = []
            done_buffer = []
            elapsed_time = time.time() - start_time
            LOGGER.info(
                f'took {time.time()-start_time:.2f}s to stitch '
                f'{n_buffered/elapsed_time:.2f} per sec into '
                f'{target_stitch_raster_path}')
            n_buffered = 0
        if payload is None:
            break


def main(watershed_ids=None):
    """Entry point.

    Args:
        watershed_ids (list): if present, only run analysis on the list
            of 'watershed,fid' strings in this list.

    Return:
        None.
    """
    LOGGER.info('create new taskgraph')
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    basename_dem = os.path.basename(os.path.splitext(DEM_ZIP_URL)[0])
    dem_download_dir = os.path.join(WORKSPACE_DIR, basename_dem)
    watershed_download_dir = os.path.join(
        WORKSPACE_DIR, os.path.basename(os.path.splitext(
            WATERSHED_VECTOR_ZIP_URL)[0]))
    population_download_dir = os.path.join(
        WORKSPACE_DIR, 'population_rasters')

    work_db_path = os.path.join(WORKSPACE_DIR, 'completed_fids.db')
    LOGGER.info('fetch completed job set')
    completed_job_set = get_completed_job_id_set(work_db_path)
    LOGGER.info(f'there are {len(completed_job_set)} completed jobs so far')

    for dir_path in [
            dem_download_dir, watershed_download_dir,
            population_download_dir]:
        os.makedirs(dir_path, exist_ok=True)

    LOGGER.info('download dem')
    download_dem_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(DEM_ZIP_URL, dem_download_dir),
        target_path_list=[
            os.path.join(
                dem_download_dir, os.path.basename(DEM_ZIP_URL))],
        task_name='download and unzip dem')

    dem_tile_dir = os.path.join(dem_download_dir, 'global_dem_3s')
    dem_vrt_path = os.path.join(
        dem_tile_dir,
        f'{os.path.basename(os.path.splitext(DEM_ZIP_URL)[0])}.vrt')
    LOGGER.debug(f'build vrt to {dem_vrt_path}')

    LOGGER.info('build vrt')
    task_graph.add_task(
        func=subprocess.run,
        args=(f'gdalbuildvrt {dem_vrt_path} {dem_tile_dir}/*.tif',),
        kwargs={'shell': True, 'check': True},
        target_path_list=[dem_vrt_path],
        dependent_task_list=[download_dem_task],
        task_name='build dem vrt')

    download_watershed_vector_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(WATERSHED_VECTOR_ZIP_URL, watershed_download_dir),
        task_name='download and unzip watershed vector')

    HAB_MASK_URL
    hab_mask_raster_path = os.path.join(
        WORKSPACE_DIR, os.path.basename(HAB_MASK_URL))
    download_hab_mask_task = task_graph.add_task(
        func=ecoshard.download_url,
        args=(HAB_MASK_URL, hab_mask_raster_path),
        target_path_list=[hab_mask_raster_path],
        task_name=f'download {HAB_MASK_URL}')

    pop_raster_path_map = {}
    stitch_raster_path_map = {}
    for pop_id, pop_url in POPULATION_RASTER_URL_MAP.items():
        pop_raster_path = os.path.join(
            population_download_dir, os.path.basename(pop_url))
        LOGGER.info(f'download {pop_url}')
        download_pop_raster = task_graph.add_task(
            func=ecoshard.download_url,
            args=(pop_url, pop_raster_path),
            target_path_list=[pop_raster_path],
            task_name=f'download {pop_url}')
        pop_raster_path_map[pop_id] = pop_raster_path
        stitch_raster_path_map[pop_id] = [
            os.path.join(WORKSPACE_DIR, f'downstream_bene_{pop_id}.tif'),
            os.path.join(WORKSPACE_DIR, f'downstream_bene_{pop_id}_normalized.tif'),
            os.path.join(WORKSPACE_DIR, f'downstream_bene_{pop_id}_hab_normalized.tif')]

        for stitch_path in stitch_raster_path_map[pop_id]:
            if not os.path.exists(stitch_path):
                driver = gdal.GetDriverByName('GTiff')
                cell_size = 10./3600. * 2  # do this for Nyquist theorem
                n_cols = int(360./cell_size)
                n_rows = int(180./cell_size)
                LOGGER.info(f'**** creating raster of size {n_cols} by {n_rows}')
                target_raster = driver.Create(
                    stitch_path,
                    int(360/cell_size), int(180/cell_size), 1,
                    gdal.GDT_Float32,
                    options=(
                        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                        'SPARSE_OK=TRUE', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
                wgs84_srs = osr.SpatialReference()
                wgs84_srs.ImportFromEPSG(4326)
                target_raster.SetProjection(wgs84_srs.ExportToWkt())
                target_raster.SetGeoTransform(
                    [-180, cell_size, 0, 90, 0, -cell_size])
                target_band = target_raster.GetRasterBand(1)
                target_band.SetNoDataValue(-9999)
                target_raster = None

    LOGGER.info('wait for downloads to conclude')
    task_graph.join()
    task_graph.close()
    task_graph = None

    apply_manager_autopatch()
    manager = multiprocessing.Manager()
    completed_work_queue = manager.Queue()
    LOGGER.info('start complete worker thread')
    global WATERSHEDS_TO_PROCESS_COUNT
    WATERSHEDS_TO_PROCESS_COUNT = 0
    # expecting 6 stitches, base, norm, habnorm times 2 pop scenarios
    job_complete_worker_thread = threading.Thread(
        target=job_complete_worker,
        args=(
            completed_work_queue, work_db_path, args.clean_result, 6))
    job_complete_worker_thread.start()

    # contains work queues for regular and normalized beneficiaries
    stitch_work_queue_list = [
        (manager.Queue(N_TO_STITCH*2),
         manager.Queue(N_TO_STITCH*2),
         manager.Queue(N_TO_STITCH*2))
        for _ in range(2)]
    stitch_worker_process_list = []
    for stitch_work_queue_tuple, target_stitch_raster_path_list in zip(
            stitch_work_queue_list,
            [stitch_raster_path_map[raster_id]
             for raster_id in POPULATION_RASTER_URL_MAP.keys()]):
        for stitch_work_queue, target_stitch_raster_path in zip(
                stitch_work_queue_tuple, target_stitch_raster_path_list):
            LOGGER.debug(f'starting a stitcher for {target_stitch_raster_path}')
            stitch_worker_process = multiprocessing.Process(
                target=stitch_worker,
                args=(
                    stitch_work_queue, target_stitch_raster_path,
                    completed_work_queue, args.clean_result))
            stitch_worker_process.start()
            stitch_worker_process_list.append(stitch_worker_process)

    watershed_work_queue = manager.Queue()

    watershed_root_dir = os.path.join(
        watershed_download_dir, 'watersheds_globe_HydroSHEDS_15arcseconds')

    if watershed_ids:
        for watershed_id in watershed_ids:
            watershed_basename, watershed_fid = watershed_id.split(',')
            watershed_path = os.path.join(
                watershed_root_dir, f'{watershed_basename}.shp')

            job_id = f'''{os.path.basename(
                os.path.splitext(watershed_path)[0])}_{watershed_fid}'''
            if job_id in completed_job_set:
                continue
            workspace_dir = os.path.join(WATERSHED_WORKSPACE_DIR, job_id)
            os.makedirs(workspace_dir, exist_ok=True)
            WATERSHEDS_TO_PROCESS_COUNT += 1
            process_watershed(
                job_id, watershed_path, int(watershed_fid), dem_vrt_path,
                hab_mask_raster_path,
                [pop_raster_path_map[raster_id]
                 for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                [os.path.join(workspace_dir, f'''downstream_benficiaries_{raster_id}_{watershed_basename}_{
                     watershed_fid}.tif''')
                 for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                [os.path.join(workspace_dir, f'''downstream_benficiaries_{raster_id}_{watershed_basename}_{
                     watershed_fid}_normalized.tif''')
                 for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                [os.path.join(workspace_dir, f'''downstream_benficiaries_{raster_id}_{watershed_basename}_{
                     watershed_fid}_hab_normalized.tif''')
                 for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                stitch_work_queue_list)
    else:
        watershed_worker_process_list = []
        for _ in range(multiprocessing.cpu_count()):
            watershed_worker_process = multiprocessing.Process(
                target=general_worker,
                args=(watershed_work_queue,))
            watershed_worker_process.start()
            watershed_worker_process_list.append(watershed_worker_process)

        LOGGER.info('building watershed fid list')
        watershed_fid_list = []
        watershed_path_list = []
        for watershed_path in glob.glob(
                os.path.join(watershed_root_dir, '*.shp')):
            watershed_path_list.append(watershed_path)
            watershed_vector = gdal.OpenEx(watershed_path, gdal.OF_VECTOR)
            watershed_layer = watershed_vector.GetLayer()
            watershed_fid_list.extend([
                (watershed_feature.GetGeometryRef().Area(),
                 watershed_feature.GetFID(), len(watershed_path_list)-1)
                for watershed_feature in watershed_layer])
            watershed_layer = None
            watershed_vector = None
        LOGGER.info('starting scheduling')
        for watershed_area, watershed_fid, watershed_path_index in sorted(
                watershed_fid_list, reverse=True):
            watershed_path = watershed_path_list[watershed_path_index]
            watershed_basename = os.path.splitext(
                os.path.basename(watershed_path))[0]
            job_id = f'''{os.path.basename(
                os.path.splitext(watershed_path)[0])}_{watershed_fid}'''
            if job_id in completed_job_set:
                continue
            WATERSHEDS_TO_PROCESS_COUNT += 1

            workspace_dir = os.path.join(WATERSHED_WORKSPACE_DIR, job_id)
            watershed_work_queue.put((
                process_watershed,
                (job_id, watershed_path, watershed_fid, dem_vrt_path,
                 hab_mask_raster_path,
                 [pop_raster_path_map[raster_id]
                  for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                 [os.path.join(workspace_dir, f'''downstream_benficiaries_{raster_id}_{watershed_basename}_{
                     watershed_fid}.tif''')
                  for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                 [os.path.join(workspace_dir, f'''downstream_benficiaries_{raster_id}_{watershed_basename}_{
                     watershed_fid}_normalized.tif''')
                  for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                 [os.path.join(workspace_dir, f'''downstream_benficiaries_{raster_id}_{watershed_basename}_{
                     watershed_fid}_hab_normalized.tif''')
                  for raster_id in POPULATION_RASTER_URL_MAP.keys()],
                 stitch_work_queue_list)))

        LOGGER.debug('waiting for watershed workers to be done')
        watershed_work_queue.put(None)
        for watershed_worker in watershed_worker_process_list:
            watershed_worker.join()
        LOGGER.debug('watershed workers are done')

    LOGGER.debug('signal stitch workers to be done')
    for stitch_queue in [
            sq for sq_tuple in stitch_work_queue_list for sq in sq_tuple]:
        stitch_queue.put(None)

    for stitch_worker_process in stitch_worker_process_list:
        stitch_worker_process.join()
    LOGGER.debug('stitch worker done')

    completed_work_queue.put(None)
    job_complete_worker_thread.join()
    LOGGER.info('compressing/overview/ecoshard result')

    hash_thread_list = []
    for pop_id in stitch_raster_path_map:
        for raster_path in stitch_raster_path_map[pop_id]:
            hash_thread = threading.Thread(
                target=hash_overview_compress_raster,
                args=(raster_path,))
            hash_thread.start()
            hash_thread_list.append(hash_thread)

    LOGGER.info('waiting for compress/overview/ecoshard complete')
    for hash_thread in hash_thread_list:
        hash_thread.join()
    LOGGER.info('all done')


def hash_overview_compress_raster(raster_path):
    """Compress, overview, then hash the raster."""
    compressed_path = '%s_compressed_overviews%s' % os.path.splitext(
        raster_path)
    ecoshard.compress_raster(
        raster_path, compressed_path, compression_algorithm='LZW')
    ecoshard.build_overviews(compressed_path)
    compressed_raster = gdal.OpenEx(
        compressed_path, gdal.OF_RASTER | gdal.GA_Update)
    compressed_raster_band = compressed_raster.GetRasterBand(1)
    stats = compressed_raster_band.ComputeStatistics(False)
    LOGGER.debug(stats)
    compressed_raster_band = None
    compressed_raster = None
    ecoshard.hash_file(compressed_path, rename=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream beneficiaries')
    parser.add_argument(
        '--watershed_ids', nargs='+',
        help='if present only run on this watershed id')
    parser.add_argument(
        '--clean_result', action='store_true',
        help='use this flag to delete the workspace after stitching')

    args = parser.parse_args()

    main(watershed_ids=args.watershed_ids)
