import numpy as np
import vigra
import os
# import cPickle as pickle
# import shutil
# import itertools
import h5py
from copy import deepcopy
from path_computation import parallel_wrapper
import logging
logger = logging.getLogger(__name__)
from joblib import Parallel,delayed

# relative imports from top level dir
from ..Postprocessing import remove_small_segments
from ..lifted_mc import compute_and_save_long_range_nh, optimize_lifted, compute_and_save_lifted_nh
from ..EdgeRF import RandomForest
from ..ExperimentSettings import ExperimentSettings
from ..tools import find_matching_row_indices

# imports from this dir
from .compute_paths_and_features import shortest_paths, distance_transform, path_feature_aggregator, \
    extract_local_graph_from_segmentation
from .compute_border_contacts import compute_path_end_pairs, compute_path_end_pairs_and_labels, \
    compute_border_contacts_old  # , compute_border_contacts

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex as nifty  # conda version build with cplex
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi as nifty  # conda version build with gurobi
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


def extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        paths_cache_folder=None, anisotropy=[1,1,10]):
    """
        extract paths from segmentation, for pipeline
    """


    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s.h5' % ds.ds_name)
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array([path.reshape((len(path) / 3, 3)) for path in all_paths])
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')

    # otherwise compute the paths
    else:
        seg = vigra.readHDF5(seg_path, key)
        dt = ds.inp(ds.n_inp - 1)
        all_paths = []
        paths_to_objs = []

        #creating distance transform of whole volume for border near paths
        volume_expanded = np.ones((dt.shape[0]+2,dt.shape[1]+2,dt.shape[1]+2))
        volume_expanded[1:-1, 1:-1, 1:-1] = 0
        volume_dt = vigra.filters.distanceTransform(
            volume_expanded.astype("uint32"), background=True,
            pixel_pitch=[10, 1, 1])[1:-1, 1:-1, 1:-1]

        #threshhold for distance transform for picking terminal
        #points near boundary
        threshhold_boundary=30
        volume_where_threshhold = np.where(volume_dt < threshhold_boundary)
        volume_dt_boundaries = np.s_[min(volume_where_threshhold[0])+1:max(volume_where_threshhold[0]),
                               min(volume_where_threshhold[1])+1:max(volume_where_threshhold[1]),
                               min(volume_where_threshhold[2])+1:max(volume_where_threshhold[2])]

        #for counting and debugging purposes
        len_uniq=len(np.unique(seg))-1

        #parallelized path computation
        parallel_array = Parallel(n_jobs=-1)\
            (delayed(parallel_wrapper)(seg, dt, [],
                                       anisotropy, label,
                                       len_uniq, volume_dt_boundaries,
                                       "only_paths")
             for label in np.unique(seg))


        [[all_paths.append(path)
           for path in seg_array[0] if seg_array!=[]]
            for seg_array in parallel_array]

        [[paths_to_objs.append(path_to_obj)
          for path_to_obj in seg_array[1] if seg_array!=[]]
            for seg_array in parallel_array]

        # all_paths=np.array(all_paths)
        paths_to_objs=np.array(paths_to_objs, dtype="float64")

        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')

    return all_paths, paths_to_objs


def extract_paths_and_labels_from_segmentation(
        ds,
        seg,
        seg_id,
        gt,
        correspondence_list,
        paths_cache_folder=None, anisotropy=[1, 1, 10]):

    """
        extract paths from segmentation, for learning
    """

    logger.debug('Extracting paths and labels from segmentation ...')

    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s_seg_%i.h5' % (ds.ds_name, seg_id))
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array([path.reshape((len(path) / 3, 3)) for path in all_paths])
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')
        path_classes = vigra.readHDF5(paths_save_file, 'path_classes')
        correspondence_list = vigra.readHDF5(paths_save_file, 'correspondence_list').tolist()

    # otherwise compute paths
    else:
        dt = ds.inp(ds.n_inp - 1)
        all_paths = []
        paths_to_objs = []
        path_classes = []

        # creating distance transform of whole volume for border near paths
        volume_expanded = np.ones((dt.shape[0] + 2, dt.shape[1] + 2, dt.shape[1] + 2))
        volume_expanded[1:-1, 1:-1, 1:-1] = 0
        volume_dt = vigra.filters.distanceTransform(
            volume_expanded.astype("uint32"), background=True,
            pixel_pitch=[10, 1, 1])[1:-1, 1:-1, 1:-1]

        # threshhold for distance transform for picking terminal
        # points near boundary
        threshhold_boundary = 30
        volume_where_threshhold = np.where(volume_dt < threshhold_boundary)
        volume_dt_boundaries = np.s_[min(volume_where_threshhold[0]) + 1:max(volume_where_threshhold[0]),
                               min(volume_where_threshhold[1]) + 1:max(volume_where_threshhold[1]),
                               min(volume_where_threshhold[2]) + 1:max(volume_where_threshhold[2])]

        # for counting and debugging purposes
        len_uniq = len(np.unique(seg)) - 1

        #parallelized path computation
        parallel_array = Parallel(n_jobs=-1)\
            (delayed(parallel_wrapper)(seg, dt, gt,
                                       anisotropy, label, len_uniq,
                                       volume_dt_boundaries)
             for label in np.unique(seg))

        [[all_paths.append(path)
           for path in seg_array[0]]
            for seg_array in parallel_array]

        [[paths_to_objs.append(path_to_obj)
          for path_to_obj in seg_array[1]]
            for seg_array in parallel_array]

        [[path_classes.append(path_class)
           for path_class in seg_array[2]]
            for seg_array in parallel_array]


        print "finished appending"
        all_paths = np.array(all_paths)
        paths_to_objs = np.array(paths_to_objs, dtype="float64")
        path_classes = np.array(path_classes)
        print "finished numpying"

        # if caching is enabled, write the results to cache
        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                logger.info('Saving paths in {}'.format(paths_save_file))
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')
            vigra.writeHDF5(path_classes, paths_save_file, 'path_classes')
            vigra.writeHDF5(correspondence_list, paths_save_file, 'correspondence_list')

    logger.debug('... done extracting paths and labels from segmentation!')

    return all_paths, paths_to_objs, \
            path_classes, correspondence_list

# cache the random forest here
def train_random_forest_for_merges(
        trainsets,  # list of datasets with training data
        mc_segs_train,  # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        paths_cache_folder=None
):

    logger.info('Training false merges random forest ...')

    rf_cache_folder = ExperimentSettings().rf_cache_folder

    if rf_cache_folder is not None:
        if not os.path.exists(rf_cache_folder):
            os.mkdir(rf_cache_folder)

    rf_save_path = '' if rf_cache_folder is None else os.path.join(
        rf_cache_folder,
        'rf_merges_%s' % '_'.join([ds.ds_name for ds in trainsets])
    )  # TODO more meaningful save name

    # check if rf is already cached
    if RandomForest.is_cached(rf_save_path):

        logger.info("Loading RF from: {}".format(rf_save_path))
        rf = RandomForest.load_from_file(rf_save_path, 'rf', ExperimentSettings().n_threads)

    # otherwise do the actual calculations
    else:

        logger.info('RF was not cached and will be computed.')

        features_train = []
        labels_train = []

        logger.debug('Looping over training sets:')

        # loop over the training datasets
        for ds_id, paths_to_betas in enumerate(mc_segs_train):

            # # FIXME delete this!
            # if ds_id == 0:
            #     continue

            logger.debug('----------')
            logger.debug('ds_id = {}'.format(ds_id))

            current_ds = trainsets[ds_id]
            logger.debug('current_ds.ds_name = {}'.format(current_ds.ds_name))

            keys_to_betas = mc_segs_train_keys[ds_id]
            assert len(keys_to_betas) == len(paths_to_betas), "%i, %i" % (len(keys_to_betas), len(paths_to_betas))

            # For training of the training data split the trainsets again
            current_trainsets = np.delete(trainsets, ds_id, axis=0).tolist()

            # Load ground truth
            gt = current_ds.gt()
            # add a fake distance transform
            # we need this to safely replace this with the actual distance transforms later
            # FIXME: Check the condition
            if current_ds.n_inp < 3:
                current_ds.add_input_from_data(np.zeros_like(gt, dtype='float32'))

            # Initialize correspondence list which makes sure that the same merge is not extracted from
            # multiple mc segmentations
            if ExperimentSettings().paths_avoid_duplicates:
                correspondence_list = []
            else:
                correspondence_list = None

            # loop over the different beta segmentations per train set
            for seg_path_id, seg_path in enumerate(paths_to_betas):
                logger.debug('seg_path_id = {}'.format(seg_path_id))
                key = keys_to_betas[seg_path_id]

                # Calculate the new distance transform and replace it in the dataset inputs
                seg = remove_small_segments(vigra.readHDF5(seg_path, key))
                dt  = distance_transform(seg, [ExperimentSettings().anisotropy_factor, 1., 1.])

                # NOTE IMPORTANT:
                # We assume that the distance transform always has the last inp_id and
                # that a (dummy) dt was already added in the beginning
                current_ds.replace_inp_from_data(current_ds.n_inp - 1, dt, clear_cache=False)
                # we delete all filters based on the distance transform
                current_ds.clear_filters(current_ds.n_inp - 1)

                # Compute the paths
                paths, paths_to_objs, path_classes, correspondence_list = extract_paths_and_labels_from_segmentation(
                    current_ds,
                    seg,
                    seg_path_id,
                    gt,
                    correspondence_list,
                    paths_cache_folder
                )

                if paths.size:
                    # TODO: decide which filters and sigmas to use here (needs to be exposed first)
                    features_train.append(path_feature_aggregator(current_ds, paths))
                    labels_train.append(path_classes)

                else:
                    print "No paths found for seg_id = {}".format(seg_path_id)
                    continue

        features_train = np.concatenate(features_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        assert features_train.shape[0] == labels_train.shape[0]
        features_train = np.nan_to_num(features_train).astype('float32')

        rf = RandomForest(
            features_train,
            labels_train,
            ExperimentSettings().n_trees,
            ExperimentSettings().n_threads
        )

        # cache the rf if caching is enabled
        if rf_cache_folder is not None:
            rf.write(rf_save_path, 'rf')

    logger.info('... done training false merges random forest!')
    return rf


def compute_false_merges(
        trainsets,  # list of datasets with training data
        ds_test,  # one dataset -> predict the false merged objects
        mc_segs_train,  # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        mc_seg_test,
        mc_seg_test_key,
        test_paths_cache_folder=None,
        train_paths_cache_folder=None
):
    """
    Computes and returns false merge candidates

    :param ds_train: Array of datasets representing multiple source images; [N x 1]
        Has to contain:
        ds_train.inp(0) := raw image
        ds_train.inp(1) := probs image
        ds_train.gt()   := groundtruth

    :param ds_test:
        Has to contain:
        ds_test.inp(0) := raw image
        ds_test.inp(1) := probs image

    :param mc_segs_train: Multiple multicut segmentations on ds_train
        Different betas for each ds_train; [N x len(betas)]
    :param mc_seg_test: Multicut segmentation on ds_test (usually beta=0.5)
    :return:
    """

    logger.info('Begin of compute_false_merges ...')

    assert len(trainsets) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"
    assert len(mc_segs_train_keys) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"

    rf = train_random_forest_for_merges(
        trainsets,
        mc_segs_train,
        mc_segs_train_keys,
        train_paths_cache_folder
    )

    # load the segmentation, compute distance transform and add it to the test dataset
    seg = vigra.readHDF5(mc_seg_test, mc_seg_test_key)
    dt = distance_transform(seg, [ExperimentSettings().anisotropy_factor, 1., 1.])
    if ds_test.n_inp < 3:
        ds_test.add_input_from_data(dt)
    else:
        ds_test.replace_inp_from_data(ds_test.n_inp - 1, dt, clear_cache=False)

    paths_test, paths_to_objs_test = extract_paths_from_segmentation(
        ds_test,
        mc_seg_test,
        mc_seg_test_key,
        test_paths_cache_folder
    )

    assert len(paths_test) == len(paths_to_objs_test)

    features_test = path_feature_aggregator(ds_test, paths_test)
    assert features_test.shape[0] == len(paths_test)
    features_test = np.nan_to_num(features_test)

    #  Cache features for debugging TODO deactivated for now
    # if not os.path.exists(paths_save_folder + '../debug'):
    #     os.mkdir(paths_save_folder + '../debug')
    # with open(paths_save_folder + '../debug/features_test.pkl', mode='w') as f:
    #     pickle.dump(features_test, f)

    return paths_test, rf.predict_probabilities(features_test)[:, 1], paths_to_objs_test


# We sample new lifted edges and save them if a cache folder is given
def sample_and_save_paths_from_lifted_edges(
        cache_folder,
        ds,
        seg,
        obj_id,
        uv_local,
        distance_transform,
        eccentricity_centers,
        reverse_mapping
):

    if cache_folder is not None:
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        save_path = os.path.join(cache_folder, 'paths_from_lifted_ds_%s_obj_%i.h5' % (ds.ds_name, obj_id))
    else:
        save_path = ''

    # check if the cache already exists
    if os.path.exists(save_path):  # if True, load paths from file
        paths_obj = vigra.readHDF5(save_path, 'paths')
        # we need to reshape the paths again to revover the coordinates
        if paths_obj.size:
            # FIXME This is a workaround to create the same type of np array even when len==1
            # FIXME I fear a similar issue when all paths have the exact same length
            if len(paths_obj) == 1:
                paths_obj = [path.reshape((len(path) / 3, 3)) for path in paths_obj]
                tmp = np.empty((1,), dtype=np.object)
                tmp[0] = paths_obj[0]
                paths_obj = tmp
            else:
                paths_obj = np.array([path.reshape((len(path) / 3, 3)) for path in paths_obj])
        uv_ids_paths_min_nh = vigra.readHDF5(save_path, 'uv_ids')

    else:  # if False, compute the paths

        # Sample uv pairs out of seg_ids (make sure to have a minimal graph dist.)
        # ------------------------------------------------------------------------
        # TODO: Alternatively sample until enough false merges are found
        uv_ids_paths_min_nh = compute_and_save_long_range_nh(
            uv_local,
            ExperimentSettings().min_nh_range,
            ExperimentSettings().max_sample_size
        )

        if uv_ids_paths_min_nh.any():
            uv_ids_paths_min_nh = np.sort(uv_ids_paths_min_nh, axis=1)

            # -------------------------------------------------------------
            # Get the distance transform of the current object

            masked_disttransf = distance_transform.copy()
            masked_disttransf[seg != obj_id] = np.inf

            # If we have a reverse mapping, turn them to the original labels
            uv_ids_paths_min_nh = np.array(
                [np.array([reverse_mapping[u] for u in uv]) for uv in uv_ids_paths_min_nh]
            )

            # Extract the respective coordinates from ecc_centers_seg thus creating pairs of coordinates
            uv_ids_paths_min_nh_coords = [[eccentricity_centers[u] for u in uv] for uv in uv_ids_paths_min_nh]

            # Compute the shortest paths according to the pairs list
            paths_obj = shortest_paths(
                masked_disttransf,
                uv_ids_paths_min_nh_coords,
                ExperimentSettings().n_threads)
            keep_mask = np.array([isinstance(x, np.ndarray) for x in paths_obj], dtype=np.bool)
            # FIXME This is a workaround to create the same type of np array even when len==1
            # FIXME I fear a similar issue when all paths have the exact same length
            if len(paths_obj) == 1:
                tmp = np.empty((1,), dtype=np.object)
                tmp[0] = paths_obj[0]
                paths_obj = tmp[keep_mask]
            else:
                paths_obj = np.array(paths_obj)[keep_mask]
            uv_ids_paths_min_nh = uv_ids_paths_min_nh[keep_mask]

        else:
            paths_obj = np.array([])

        # cache the paths if we have caching activated
        if cache_folder is not None:
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            paths_save = np.array([pp.flatten() for pp in paths_obj])
            try:
                # need to write paths with vlen and flatten before writing to properly save this
                with h5py.File(save_path) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(paths_save[0].dtype))
                    f.create_dataset('paths', data=paths_save, dtype=dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(paths_save, save_path, 'paths')

            vigra.writeHDF5(uv_ids_paths_min_nh, save_path, 'uv_ids')

    return paths_obj, uv_ids_paths_min_nh


# combine sampled and extra paths
def combine_paths(
        paths_obj,
        extra_paths,
        uv_ids_paths_min_nh,
        seg,
        mapping=None
):

    # find coordinates belonging to the extra paths
    extra_coords = [[tuple(p[0]), tuple(p[-1])] for p in extra_paths]

    # map coordinates to uv ids
    if mapping is None:
        extra_path_uvs = np.array([np.array(
            [seg[coord[0]],
             seg[coord[1]]]) for coord in extra_coords])
    else:
        extra_path_uvs = np.array([np.array(
            [mapping[seg[coord[0]]],
             mapping[seg[coord[1]]]]) for coord in extra_coords])
    extra_path_uvs = np.sort(extra_path_uvs, axis=1)

    # exclude paths with u == v
    different_uvs = extra_path_uvs[:, 0] != extra_path_uvs[:, 1]
    extra_path_uvs = extra_path_uvs[different_uvs]
    extra_paths = extra_paths[different_uvs]

    # concatenate exta paths and sampled paths (modulu duplicates)
    if uv_ids_paths_min_nh.any():  # only concatenate if we have sampled paths
        matches = find_matching_row_indices(uv_ids_paths_min_nh, extra_path_uvs)
        if matches.size:  # if we have matching uv ids, exclude them from the extra paths before concatenating
            duplicate_mask = np.ones(len(extra_path_uvs), dtype=np.bool)
            duplicate_mask[matches[:, 1]] = False
            extra_path_uvs = extra_path_uvs[duplicate_mask]
            extra_paths = extra_paths[duplicate_mask]
        return np.concatenate([paths_obj, extra_paths]), np.concatenate([uv_ids_paths_min_nh, extra_path_uvs], axis=0)

    else:
        return extra_paths, extra_path_uvs


# resolve each potential false merge individually with lifted edges
def resolve_merges_with_lifted_edges(
        ds,
        train_sets,
        seg_id,
        false_paths,  # dict(merge_ids : false_paths)
        path_rf,
        mc_segmentation,
        mc_weights_all,  # the precomputed mc-weights
        paths_cache_folder=None,
        lifted_weights_all=None  # pre-computed lifted mc-weights
):
    assert isinstance(false_paths, dict)

    # NOTE: We assume that the dataset already has a distance transform added as last input
    # This should work out, because we have already detected false merge paths for this segmentation
    disttransf = ds.inp(ds.n_inp - 1)
    # Pre-processing of the distance transform
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path
    #    detection) should be at the center of the current process
    disttransf = np.amax(disttransf) - disttransf
    #
    # c) Increase the value difference between pixels near the boundaries and pixels
    #    central within the processes. This increases the likelihood of the paths to
    #    follow the center of processes, thus avoiding short-cuts
    disttransf = np.power(disttransf, ExperimentSettings().paths_penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume

    # I have moved this to the dataset to have it cached
    ecc_centers_seg = ds.eccentricity_centers(seg_id, True)

    # get local and lifted uv ids
    uv_ids = ds.uv_ids(seg_id)
    uv_ids_lifted = compute_and_save_lifted_nh(
        ds,
        seg_id,
        ExperimentSettings().lifted_neighborhood,
        False
    )

    # iterate over the obj-ids which have a potential false merge
    # for each, sample new lifted edges and resolve the obj individually
    resolved_objs = {}
    for merge_id in false_paths:

        local_uv_mask, lifted_uv_mask, mapping, reverse_mapping = extract_local_graph_from_segmentation(
            ds,
            seg_id,
            mc_segmentation,
            merge_id,
            uv_ids,
            uv_ids_lifted
        )

        # extract local uv ids and corresponding weights
        uv_local = np.array([[mapping[u] for u in uv] for uv in uv_ids[local_uv_mask]])
        mc_weights     = mc_weights_all[local_uv_mask]

        # extract the lifted uv ids and corresponding weights
        uv_local_lifted = np.array([[mapping[u] for u in uv] for uv in uv_ids_lifted[lifted_uv_mask]])
        lifted_weights = lifted_weights_all[lifted_uv_mask]

        # FIXME somthing goes wrong here, but what? merge_id = 75
        # FIXME ecc_centers_seg still doesn't have switched dimensions!!! -> Old cache or in computation?
        # sample new paths corresponding to lifted edges with min graph distance
        paths_obj, uv_ids_paths_min_nh = sample_and_save_paths_from_lifted_edges(
            paths_cache_folder,
            ds,
            mc_segmentation,
            merge_id,
            uv_local,
            disttransf,
            ecc_centers_seg,
            reverse_mapping
        )

        # Map to local uvs
        uv_ids_paths_min_nh = np.array([[mapping[u] for u in uv] for uv in uv_ids_paths_min_nh])

        # add the paths that were initially classified
        paths_obj, uv_ids_paths_min_nh = combine_paths(
            paths_obj,
            np.array(false_paths[merge_id]),  # <- initial paths
            uv_ids_paths_min_nh,
            seg,
            mapping)

        if not paths_obj.size:
            continue

        # Compute the path features
        features = path_feature_aggregator(ds, paths_obj)
        features = np.nan_to_num(features)

        # Cache features for debug purpose # TODO disabled for now
        # with open(export_paths_path + '../debug/features_resolve_{}.pkl'.format(merge_id), mode='w') as f:
        #    pickle.dump(features, f)

        # compute the lifted weights from rf probabilities
        # FIXME TODO - not caching this for now -> should not be performance relevant
        lifted_path_weights = path_rf.predict_probabilities(features)[:, 1]

        # Class 1: contain a merge
        # Class 0: don't contain a merge

        # scale the probabilities
        p_min = 0.001
        p_max = 1. - p_min
        lifted_path_weights = (p_max - p_min) * lifted_path_weights + p_min

        # Transform probs to weights
        lifted_path_weights = np.log((1 - lifted_path_weights) / lifted_path_weights)

        # Weighting edges with their length for proper lifted to local scaling
        lifted_path_weights /= lifted_path_weights.shape[0]
        lifted_path_weights *= ExperimentSettings().lifted_path_weights_factor
        lifted_weights /= lifted_weights.shape[0]
        mc_weights /= mc_weights.shape[0]

        # Concatenate all lifted weights and edges
        if lifted_weights.size:  # only concatenate if we have lifted edges from sampling
            lifted_weights = np.concatenate(
                (lifted_path_weights, lifted_weights),
                axis=0
            )
            uv_ids_lifted_nh_total = np.concatenate(
                (uv_ids_paths_min_nh, uv_local_lifted),
                axis=0
            )
        else:
            lifted_weights = lifted_path_weights
            uv_ids_lifted_nh_total = uv_ids_paths_min_nh

        resolved_nodes, _, _ = optimize_lifted(
            uv_local,
            uv_ids_lifted_nh_total,
            mc_weights,
            lifted_weights
        )

        resolved_nodes, _, _ = vigra.analysis.relabelConsecutive(resolved_nodes, start_label=0, keep_zeros=False)
        # project back to global node ids and save
        resolved_objs[merge_id] = {reverse_mapping[i]: node_res for i, node_res in enumerate(resolved_nodes)}

    return resolved_objs


def resolve_merges_with_lifted_edges_global(
        ds,
        seg_id,
        false_paths,  # dict(merge_ids : false_paths)
        path_rf,
        mc_segmentation,
        mc_weights_all,  # the precomputed mc-weights
        paths_cache_folder=None,
        lifted_weights_all=None  # pre-computed lifted mc-weights
):
    assert isinstance(false_paths, dict)

    # NOTE: We assume that the dataset already has a distance transform added as last input
    # This should work out, because we have already detected false merge paths for this segmentation
    disttransf = ds.inp(ds.n_inp - 1)
    # Pre-processing of the distance transform
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path
    #    detection) should be at the center of the current process
    disttransf = np.amax(disttransf) - disttransf
    #
    # c) Increase the value difference between pixels near the boundaries and pixels
    #    central within the processes. This increases the likelihood of the paths to
    #    follow the center of processes, thus avoiding short-cuts
    disttransf = np.power(disttransf, ExperimentSettings().paths_penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume

    # I have moved this to the dataset to have it cached
    ecc_centers_seg = ds.eccentricity_centers(seg_id, True)

    # get local and lifted uv ids
    uv_ids = ds.uv_ids(seg_id)
    uv_ids_lifted = compute_and_save_lifted_nh(
        ds,
        seg_id,
        ExperimentSettings().lifted_neighborhood,
        False
    )

    lifted_path_weights_all = []
    uv_ids_paths_min_nh_all = []

    for merge_id in false_paths:

        local_uv_mask, mapping, reverse_mapping = extract_local_graph_from_segmentation(
            ds,
            seg_id,
            mc_segmentation,
            merge_id,
            uv_ids
        )

        uv_ids_in_obj_local = np.array([[mapping[u] for u in uv] for uv in uv_ids[local_uv_mask]])

        # sample new paths corresponding to lifted edges with min graph distance
        paths_obj, uv_ids_paths_min_nh = sample_and_save_paths_from_lifted_edges(
            paths_cache_folder,
            ds,
            mc_segmentation,
            merge_id,
            uv_ids_in_obj_local,
            disttransf,
            ecc_centers_seg,
            reverse_mapping=reverse_mapping
        )

        # add the paths that were initially classified
        paths_obj, uv_ids_paths_min_nh = combine_paths(
            paths_obj,
            np.array(false_paths[merge_id]),  # <- initial paths
            uv_ids_paths_min_nh,
            seg)

        if not paths_obj.size:
            continue

        # Compute the path features
        features = path_feature_aggregator(ds, paths_obj)
        features = np.nan_to_num(features)

        # Cache features for debug purpose # TODO not caching for now
        # with open(export_paths_path + '../debug/features_resolve_{}.pkl'.format(merge_id), mode='w') as f:
        #    pickle.dump(features, f)

        # compute the lifted weights from rf probabilities
        lifted_path_weights = path_rf.predict_probabilities(features)[:, 1]

        # Class 1: contain a merge
        # Class 0: don't contain a merge

        # scale the probabilities
        p_min = 0.001
        p_max = 1. - p_min
        lifted_path_weights = (p_max - p_min) * lifted_path_weights + p_min

        # Transform probs to weights
        lifted_path_weights = np.log((1 - lifted_path_weights) / lifted_path_weights)

        lifted_path_weights_all.append(lifted_path_weights)
        uv_ids_paths_min_nh_all.append(uv_ids_paths_min_nh)

    lifted_path_weights_all = np.concatenate(lifted_path_weights_all)
    uv_ids_paths_min_nh_all = np.concatenate(uv_ids_paths_min_nh_all)

    # Weighting edges with their length for proper lifted to local scaling
    lifted_path_weights_all /= lifted_path_weights_all.shape[0] * ExperimentSettings().lifted_path_weights_factor
    lifted_weights_all /= lifted_weights_all.shape[0]
    mc_weights_all /= mc_weights_all.shape[0]

    # Concatenate all lifted weights and edges
    lifted_weights = np.concatenate(
        [lifted_path_weights_all, lifted_weights_all],
        axis=0
    )
    all_uv_ids_lifted_nh_total = np.concatenate(
        [uv_ids_paths_min_nh_all, uv_ids_lifted],
        axis=0
    )

    resolved_nodes, _, _ = optimize_lifted(
        uv_ids,
        all_uv_ids_lifted_nh_total,
        mc_weights_all,
        lifted_weights
    )

    resolved_nodes, _, _ = vigra.analysis.relabelConsecutive(resolved_nodes, start_label=0, keep_zeros=False)
    assert len(resolved_nodes) == uv_ids.max() + 1
    return resolved_nodes


def project_resolved_objects_to_segmentation(
        ds,
        seg_id,
        mc_segmentation,
        resolved_objs
):

    n_threads = ExperimentSettings().n_threads
    rag = ds.rag(seg_id)
    gt  = ds.gt()
    # recover the node labeling from the segmentation
    mc_labeling = nrag.gridRagAccumulateLabels(rag, gt, n_threads)

    # offset for new labels
    new_label_offset = np.max(mc_labeling) + 1

    # iterate over the resolved objs and insert their solution into
    # the node labeling
    for obj in resolved_objs:
        resolved_nodes = resolved_objs[obj]
        for node_id in resolved_nodes:
            mc_labeling[node_id] = new_label_offset + resolved_nodes[node_id]
        new_label_offset += np.max(resolved_nodes.values()) + 1

    # make consecutive and project back to segmentation
    mc_labeling = vigra.analysis.relabelConsecutive(mc_labeling, start_label=1, keep_zeros=False)
    return nrag.projectScalarNodeDataToPixels(rag, mc_labeling, n_threads)
