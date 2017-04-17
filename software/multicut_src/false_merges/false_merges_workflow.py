from compute_paths_and_features import shortest_paths, distance_transform
from multicut_src import probs_to_energies
from multicut_src import remove_small_segments
from multicut_src import compute_and_save_long_range_nh, optimizeLifted
from multicut_src import compute_and_save_lifted_nh
from multicut_src import learn_and_predict_rf_from_gt
from multicut_src import ExperimentSettings
# from find_false_merges_src import path_features_from_feature_images
# from find_false_merges_src import path_classification
from compute_paths_and_features import path_feature_aggregator
from multicut_src.tools import cache_name
from compute_border_contacts import compute_path_end_pairs, compute_path_end_pairs_and_labels, compute_border_contacts

import numpy as np
import vigra
import os
import cPickle as pickle
import shutil
import itertools
from copy import deepcopy


def extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        params):

    # TODO we don't remove small objects for now, because this would relabel the segmentation, which we don't want in this case
    seg = vigra.readHDF5(seg_path, key)
    dt = ds.inp_id(ds.n_inp-1) # we assume that the last input is the distance transform

    # Compute path end pairs
    # TODO parallelize this function !
    border_contacts = compute_border_contacts(seg, dt)
    path_pairs, paths_to_objs = compute_path_end_pairs(border_contacts)
    # Sort the paths_to_objs by size (not doing that leads to a possible bug in the next loop)
    order = np.argsort(paths_to_objs)
    paths_to_objs = np.array(paths_to_objs)[order].tolist()
    path_pairs = np.array(path_pairs)[order].tolist()

    # Invert the distance transform and take penalty power
    dt = np.amax(dt) - dt
    dt = np.power(dt, 10)

    all_paths = []
    for obj in np.unique(paths_to_objs):

        # Mask distance transform to current object
        masked_dt = deepcopy(dt)
        masked_dt[seg != obj] = np.inf

        # Take only the relevant path pairs
        pairs_in = np.array(path_pairs)[np.where(np.array(paths_to_objs) == obj)[0]]

        paths = shortest_paths(masked_dt, pairs_in, n_threads = 32)
        # paths is now a list of numpy arrays
        all_paths.extend(paths)

    # # compute the actual paths
    # all_paths = shortest_paths(dt, path_pairs, n_threads = 20)

    # Remove all paths that are None, i.e. were initially not computed or were subsequently removed
    keep_mask = [x is not None for x in all_paths]
    keep_indices = np.where(keep_mask)[0]
    all_paths = np.array(all_paths)[keep_indices].tolist()
    paths_to_objs = np.array(paths_to_objs)[keep_indices].tolist()

    return all_paths, paths_to_objs


# TODO refactor params
def extract_paths_and_labels_from_segmentation(
        ds,
        seg,
        params,
        gt,
        correspondence_list):
    """
    params:
    """

    assert seg.shape == gt.shape
    dt = ds.inp_id(ds.n_inp-1) # we assume that the last input is the distance transform

    # Compute path end pairs
    # TODO parallelize this function !
    border_contacts = compute_border_contacts(seg, dt)
    # This is supposed to only return those pairs that will be used for path computation
    # TODO: Throw out some under certain conditions (see also within function)
    path_pairs, paths_to_objs, path_classes, path_gt_labels, correspondence_list = compute_path_end_pairs_and_labels(
        border_contacts, gt, correspondence_list
    )

    # Invert the distance transform and take penalty power
    dt = np.amax(dt) - dt
    dt = np.power(dt, 10)

    all_paths = []
    for obj in np.unique(paths_to_objs):

       # Mask distance transform to current object
       masked_dt = deepcopy(dt)
       masked_dt[seg != obj] = np.inf

       # Take only the relevant path pairs
       pairs_in = np.array(path_pairs)[np.where(np.array(paths_to_objs) == obj)[0]]

       paths = shortest_paths(masked_dt, pairs_in, n_threads = 32)
       # paths is now a list of numpy arrays
       all_paths.extend(paths)

    # TODO: Here we have to ensure that every path is actually computed
    # TODO:  --> Throw not computed paths out of the lists

    # TODO: Remove paths under certain criteria
    # TODO: Do this only if GT is supplied
    # a) Class 'non-merged': Paths cross labels in GT multiple times
    # b) Class 'merged': Paths have to contain a certain amount of pixels in both GT classes
    # TODO implement stuff here

    # Remove all paths that are None, i.e. were initially not computed or were subsequently removed
    keep_mask = [x is not None for x in all_paths]
    keep_indices = np.where(keep_mask)[0]
    all_paths = np.array(all_paths)[keep_indices].tolist()
    paths_to_objs = np.array(paths_to_objs)[keep_indices].tolist()
    path_classes = np.array(path_classes)[keep_indices].tolist()

    return all_paths, paths_to_objs, path_classes, correspondence_list


# TODO move all training related stuff here
# cache the random forest here
def train_random_forest_for_merges(
        trainsets, # list of datasets with training data
        mc_segs_train, # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        params,
        rf_save_folder=None,
        paths_save_folder=None
):

    # TODO use vigra rf 3 once it is in the vigra conda package
    from sklearn.ensemble import RandomForestClassifier as Skrf

    caching = False
    if rf_save_folder != None:
        caching = True
        if not os.path.exists(rf_save_folder):
            os.mkdir(rf_save_folder)
    if paths_save_folder != None:
        if not os.path.exists(paths_save_folder):
            os.mkdir(paths_save_folder)

    rf_save_path = None if rf_save_folder == None else os.path.join(
        rf_save_folder,
        'rf_merges_%s.pkl' % '_'.join([ds.ds_name for ds in trainsets])
    ) # TODO more meaningful save name


    # check if the rf will be cached and if yes, if it is already cached
    if caching and os.path.exists(rf_save_path):
        print "Loading rf from:", rf_save_path
        with open(rf_save_path) as f:
            rf = pickle.load(f)

    # otherwise do the actual calculations
    else:

        print rf_save_path
        features_train = []
        labels_train = []

        # loop over the training datasets
        for ds_id, paths_to_betas in enumerate(mc_segs_train):

            all_paths = []
            all_paths_to_objs = []
            all_path_classes = []

            paths_save_path = None if paths_save_folder == None else os.path.join(
                paths_save_folder,
                'path_%s.pkl' % '_'.join([trainsets[ds_id].ds_name])
            )

            # TODO check if we still need this caching monstrosity once the paths are sped up
            # if we still do, this needs to be refactored!
            cached_paths = []
            print "Looking for paths folder: {}".format(paths_save_path)
            if caching and os.path.exists(paths_save_path):
                # If the paths already exist (necessary if new features should be used)
                print "Loading paths from:", paths_save_path
                with open(paths_save_path, mode='r') as f:
                    cached_paths = pickle.load(f)

            current_ds = trainsets[ds_id]
            keys_to_betas = mc_segs_train_keys[ds_id]
            assert len(keys_to_betas) == len(paths_to_betas), "%i, %i" % (len(keys_to_betas), len(paths_to_betas))

            # Load ground truth
            gt = current_ds.gt()
            # add a fake distance transform
            # we need this to safely replace this with the actual distance transforms later
            current_ds.add_input_from_data(np.zeros_like(gt.shape))

            # Initialize correspondence list which makes sure that the same merge is not extracted from
            # multiple mc segmentations
            if params.paths_avoid_duplicates:
                correspondence_list = []
            else:
                correspondence_list = None

            # loop over the different beta segmentations per train set
            for seg_id, seg_path in enumerate(paths_to_betas):
                key = keys_to_betas[seg_id]

                # Calculate the new distance transform and replace it in the dataset inputs
                seg = remove_small_segments(vigra.readHDF5(seg_path, key))
                dt  = distance_transform(seg, [1.,1.,params.anisotropy_factor])
                # NOTE IMPORTANT: We assume that the distance transform always has the last inp_id and that a (dummy) dt was already added in the beginning
                ds.replace_inp_from_data(ds.n_inp - 1, dt, clear_cache = False)
                # we delete all filters based on the distance transform
                ds.clear_filters(ds.n_inp - 1)

                if not cached_paths:
                    # Compute the paths
                    paths, paths_to_objs, path_classes, correspondence_list = extract_paths_and_labels_from_segmentation(
                            current_ds,
                            seg,
                            params,
                            gt,
                            correspondence_list)

                else:

                    # Get the paths and stuff for the current object
                    paths = cached_paths['paths'][seg_id]
                    paths_to_objs = cached_paths['paths_to_objs'][seg_id]
                    path_classes = cached_paths['path_classes'][seg_id]

                all_paths.append(paths)
                all_paths_to_objs.append(paths_to_objs)
                all_path_classes.append(path_classes)

                if paths:

                    # TODO: Extract features from paths
                    # TODO: decide which filters and sigmas to use here (needs to be exposed first)
                    features_train.append(
                        path_feature_aggregator(current_ds, paths, params)
                    )
                    labels_train.append(path_classes)

                else:

                    print "No paths found for seg_id = {}".format(seg_id)

            if not cached_paths and caching:
                print "Saving paths to:", paths_save_path
                with open(paths_save_path, 'w') as f:
                    pickle.dump(
                        {
                            'paths': all_paths,
                            'paths_to_objs': all_paths_to_objs,
                            'path_classes': all_path_classes
                        }, f
                    )

        features_train = np.concatenate(features_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        assert features_train.shape[0] == labels_train.shape[0]

        # remove nans
        features_train = np.nan_to_num(features_train).astype('float32')

        # TODO vigra.rf3
        # TODO set n_threads from global param object
        n_threads = 8
        rf = Skrf(n_jobs = n_threads)
        rf.fit(features_train, labels_train)
        if caching:
            print "Saving path-rf to:", rf_save_path
            with open(rf_save_path, 'w') as f:
                pickle.dump(rf, f)

    return rf


def compute_false_merges(
        trainsets, # list of datasets with training data
        ds_test, # one dataset -> predict the false merged objects
        mc_segs_train, # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        mc_seg_test,
        mc_seg_test_key,
        rf_save_folder = None,
        paths_save_folder=None,
        train_paths_save_folder=None,
        params=ExperimentSettings()
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
    :param params:
    :return:
    """

    assert len(trainsets) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"
    assert len(mc_segs_train_keys) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"

    # TODO: Store results of each step???

    rf = train_random_forest_for_merges(
        trainsets,
        mc_segs_train,
        mc_segs_train_keys,
        #gtruths,
        #gtruths_keys,
        params,
        rf_save_folder,
        train_paths_save_folder
    )

    paths_save_path = None if paths_save_folder == None else os.path.join(
        paths_save_folder,
        'path_{}.pkl'.format(ds_test.ds_name)
    )
    caching = False
    if paths_save_folder is not None:
        caching = True
    cached_paths = []
    if caching and os.path.exists(paths_save_path):
        # If the paths already exist (necessary if new features should be used)
        print "Loading paths from:", paths_save_path
        with open(paths_save_path, mode='r') as f:
            cached_paths = pickle.load(f)

        paths_test = cached_paths['paths']
        paths_to_objs_test = cached_paths['paths_to_objs']

        # load the segmentation, compute distance transform and add it to the test dataset
        seg = vigra.readHDF5(mc_seg_test, mc_seg_test_key)
        dt = distance_transform(seg, [1., 1., params.anisotropy_factor])
        ds_test.add_input_from_data(dt)

    else:
        paths_test, paths_to_objs_test = extract_paths_from_segmentation(
            ds_test,
            mc_seg_test,
            mc_seg_test_key,
            params
        )
        if caching:
            with open(paths_save_path, 'w') as f:
                pickle.dump(
                    {
                        'paths': paths_test,
                        'paths_to_objs': paths_to_objs_test,
                    }, f
                )

    assert len(paths_test) == len(paths_to_objs_test)

    features_test = path_feature_aggregator(
            ds_test,
            paths_test,
            params)
    assert features_test.shape[0] == len(paths_test)
    features_test = np.nan_to_num(features_test)
    # TODO vigra.rf3
    # We keep the second channel as we are looking for paths crossing a merging site (class = 1)
    # FIXME Remove this
    # Cache features for debugging
    if not os.path.exists(paths_save_folder + '../debug'):
        os.mkdir(paths_save_folder + '../debug')
    with open(paths_save_folder + '../debug/features_test.pkl', mode='w') as f:
        pickle.dump(features_test, f)
    return paths_test, rf.predict_proba(features_test)[:,1], paths_to_objs_test


# FIXME:
# resolve_merges_with_lifted_edges_global and
# resolve_merges_with_lifted_edges
# copy a lot of code that could be refactored to a single function

def resolve_merges_with_lifted_edges(
        ds,
        seg_id,
        false_paths, # dict(merge_ids : false_paths)
        path_rf,
        mc_segmentation,
        mc_weights_all, # the precomputed mc-weights
        exp_params,
        export_paths_path=None,
        lifted_weights_all=None # pre-computed lifted mc-weights
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
    disttransf = np.power(disttransf, exp_params.paths_penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume

    # I have moved this to the dataset to have it cached
    ecc_centers_seg = ds.eccentricity_centers(seg_id, True)

    # get the region adjacency graph
    rag = ds._rag(seg_id)

    # get the multicut weights
    uv_ids = rag.uvIds()

    # Get the lifted nh of the full segmentation
    uv_ids_lifted = compute_and_save_lifted_nh(
        ds,
        seg_id,
        exp_params.lifted_neighborhood,
        False
    )

    if export_paths_path is not None:
        if not os.path.exists(export_paths_path):
            os.mkdir(export_paths_path)

    resolved_objs = {}
    for merge_id in false_paths:

        mask = mc_segmentation == merge_id
        seg_ids = np.unique(seg[mask])

        # Extract the sub graph mc problem
        compare = np.in1d(uv_ids, seg_ids)
        compare = compare.reshape(uv_ids.shape).all(axis = 1)
        #compare = np.swapaxes(np.reshape(compare, uv_ids.shape), 0, 1)
        #compare = np.logical_and(compare[0], compare[1])
        mc_weights = mc_weights_all[compare]

        compare_list = list(itertools.compress(xrange(len(compare)), np.logical_not(compare)))
        uv_ids_in_seg = np.delete(uv_ids, compare_list, axis=0)

        # FIXME this does not work if lifted_weights_all are none!
        # Extract the sub graph lifted mc problem
        uv_mask = np.in1d(uv_ids_lifted, seg_ids)
        uv_mask = uv_mask.reshape(uv_ids_lifted.shape).all(axis = 1)
        #uv_mask = np.swapaxes(np.reshape(uv_mask, uv_ids_lifted.shape), 0, 1)
        #uv_mask = np.logical_and(uv_mask[0], uv_mask[1])
        lifted_weights = lifted_weights_all[uv_mask]

        ids_in_mask = list(itertools.compress(xrange(len(uv_mask)), np.logical_not(uv_mask)))
        uv_ids_lifted_in_seg = np.delete(uv_ids_lifted, ids_in_mask, axis=0)

        # Now map the uv ids to locally consecutive ids
        # local graph (consecutive in obj)
        seg_ids_local, _, mapping = vigra.analysis.relabelConsecutive(seg_ids, start_label=0, keep_zeros=False)

        # mapping = old to new,
        # reverse = new to old
        reverse_mapping = {val: key for key, val in mapping.iteritems()}
        # edge dict
        uv_local = np.array([[mapping[u] for u in uv] for uv in uv_ids_in_seg])
        uv_local_lifted = np.array([[mapping[u] for u in uv] for uv in uv_ids_lifted_in_seg])

        # Next we want to introduce the lifted path edges
        if export_paths_path is None or not os.path.isfile(
                        export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id)):

            # Sample uv pairs out of seg_ids (make sure to have a minimal graph dist.)
            # ------------------------------------------------------------------------
            # TODO: Alternatively sample until enough false merges are found
            min_range = exp_params.min_nh_range
            max_sample_size = exp_params.max_sample_size
            uv_ids_paths_min_nh_local = compute_and_save_long_range_nh(
                uv_local,
                min_range,
                max_sample_size
            )

            if uv_ids_paths_min_nh_local.any():
                uv_ids_paths_min_nh_local = np.sort(uv_ids_paths_min_nh_local, axis = 1)

                # -------------------------------------------------------------
                # Get the distance transform of the current object

                masked_disttransf = deepcopy(disttransf)
                masked_disttransf[np.logical_not(mask)] = np.inf

                # Turn them to the original labels
                uv_ids_paths_min_nh = np.array([ np.array([reverse_mapping[u] for u in uv]) for uv in uv_ids_paths_min_nh_local])

                # Extract the respective coordinates from ecc_centers_seg thus creating pairs of coordinates
                uv_ids_paths_min_nh_coords = [[ecc_centers_seg[u] for u in uv] for uv in uv_ids_paths_min_nh]

                # Compute the shortest paths according to the pairs list
                paths_obj = shortest_paths(
                    masked_disttransf,
                    uv_ids_paths_min_nh_coords,
                    32) # TODO set n_threads from global params

            else:
                paths_obj = []

            # add the paths actually classified as being wrong if not already present
            extra_paths = false_paths[merge_id]
            # first we map them to segments
            extra_coords = [[tuple(p[0]), tuple(p[-1])] for p in extra_paths]
            extra_path_uvs = np.array([np.array(
                [mapping[seg[coord[0]]],
                 mapping[seg[coord[1]]]]) for coord in extra_coords])
            extra_path_uvs = np.sort(extra_path_uvs, axis=1)
            # Extra path uv pairs have to be different
            are_different = np.not_equal(extra_path_uvs[:, 0], extra_path_uvs[:, 1])
            extra_path_uvs = extra_path_uvs[are_different, :]
            extra_paths = extra_paths[are_different]

            if uv_ids_paths_min_nh_local.any():
                for extra_id, extra_uv in enumerate(extra_path_uvs):
                    if not any(np.equal(uv_ids_paths_min_nh_local, extra_uv).all(
                            1)):  # check whether this uv - pair is already present
                        paths_obj.append(extra_paths[extra_id])
                        uv_ids_paths_min_nh_local = np.append(uv_ids_paths_min_nh_local, extra_uv[None, :], axis=0)
            else:
                paths_obj = extra_paths.tolist()
                uv_ids_paths_min_nh_local = extra_path_uvs

        else:
            # Load paths
            print 'Loading lifted edges paths from: {}'.format(
                export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id)
            )
            with open(export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id), mode='r') as f:
                paths_obj = pickle.load(f)

            # We have to get the local uv ids of the loaded paths
            # This is a shorter version of the above
            extra_paths = paths_obj
            # first we map them to segments
            extra_coords = [[tuple(p[0]), tuple(p[-1])] for p in extra_paths]
            extra_path_uvs = np.array([np.array(
                [mapping[seg[coord[0]]],
                 mapping[seg[coord[1]]]]) for coord in extra_coords])
            extra_path_uvs = np.sort(extra_path_uvs, axis=1)
            uv_ids_paths_min_nh_local = np.empty((0, 2), extra_path_uvs.dtype)
            for extra_id, extra_uv in enumerate(extra_path_uvs):
                uv_ids_paths_min_nh_local = np.append(uv_ids_paths_min_nh_local, extra_uv[None, :], axis=0)

        if paths_obj:

            if export_paths_path is None or not os.path.isfile(export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id)):

                # Compute the path features
                features = path_feature_aggregator(ds, paths_obj, exp_params)
                features = np.nan_to_num(features)

                # Cache features for debug purpose
                with open(export_paths_path + '../debug/features_resolve_{}.pkl'.format(merge_id), mode='w') as f:
                    pickle.dump(features, f)

                # fs = path_feature_aggregator(ds, (extra_paths[0],), exp_params)
                # with open(export_paths_path + '../debug/fs_{}.pkl'.format(merge_id), mode='w') as f:
                #     pickle.dump(fs, f)

                # compute the lifted weights from rf probabilities
                lifted_path_weights = path_rf.predict_proba(features)[:,1]

                # Cache paths for evaluation purposes
                if export_paths_path is not None:
                    with open(export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id), mode='w') as f:
                        pickle.dump(paths_obj, f)
                    with open(export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id), mode='w') as f:
                        pickle.dump(lifted_path_weights, f)

            else:
                # Load path probabilities
                print 'Loading lifted edges path weights from: {}'.format(
                    export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id)
                )
                with open(export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id), mode='r') as f:
                    lifted_path_weights = pickle.load(f)

            # Class 1: contain a merge
            # Class 0: don't contain a merge

            # scale the probabilities
            p_min = 0.001
            p_max = 1. - p_min
            lifted_path_weights = (p_max - p_min) * lifted_path_weights + p_min

            # Transform probs to weights
            lifted_path_weights = np.log((1 - lifted_path_weights) / lifted_path_weights)

            # Weighting edges with their length for proper lifted to local scaling
            lifted_path_weights /= lifted_path_weights.shape[0] * exp_params.lifted_path_weights_factor
            lifted_weights /= lifted_weights.shape[0]
            mc_weights /= mc_weights.shape[0]

            # Concatenate all lifted weights and edges
            # FIXME this does not work if lifted_weights_all are none!
            lifted_weights = np.concatenate(
                (lifted_path_weights, lifted_weights),
                axis=0 # TODO check for correct axis
            )
            uv_ids_lifted_nh_total = np.concatenate(
                (uv_ids_paths_min_nh_local, uv_local_lifted),
                axis=0 # TODO check for correct axis
            )

            resolved_nodes = optimizeLifted(
                uv_local,
                uv_ids_lifted_nh_total,
                mc_weights,
                lifted_weights
            )

            resolved_nodes, _, _ = vigra.analysis.relabelConsecutive(resolved_nodes, start_label = 0, keep_zeros = False)
            # project back to global node ids and save
            resolved_objs[merge_id] = {reverse_mapping[i] : node_res for i, node_res in enumerate(resolved_nodes)}

    return resolved_objs


def resolve_merges_with_lifted_edges_global(
        ds,
        seg_id,
        false_paths, # dict(merge_ids : false_paths)
        path_rf,
        mc_segmentation,
        mc_weights_all, # the precomputed mc-weights
        exp_params,
        export_paths_path=None,
        lifted_weights_all=None # pre-computed lifted mc-weights
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
    disttransf = np.power(disttransf, exp_params.paths_penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume

    # I have moved this to the dataset to have it cached
    ecc_centers_seg = ds.eccentricity_centers(seg_id, True)

    # get the region adjacency graph
    rag = ds._rag(seg_id)

    # get the multicut weights
    uv_ids = rag.uvIds()

    # Get the lifted nh of the full segmentation
    uv_ids_lifted = compute_and_save_lifted_nh(
        ds,
        seg_id,
        exp_params.lifted_neighborhood,
        False
    )

    if export_paths_path is not None:
        if not os.path.exists(export_paths_path):
            os.mkdir(export_paths_path)

    resolved_objs = {}
    for merge_id in false_paths:

        mask = mc_segmentation == merge_id
        seg_ids = np.unique(seg[mask])

        # Extract the sub graph mc problem
        compare = np.in1d(uv_ids, seg_ids)
        compare = compare.reshape(uv_ids.shape).all(axis = 1)
        #compare = np.swapaxes(np.reshape(compare, uv_ids.shape), 0, 1)
        #compare = np.logical_and(compare[0], compare[1])
        mc_weights = mc_weights_all[compare]

        compare_list = list(itertools.compress(xrange(len(compare)), np.logical_not(compare)))
        uv_ids_in_seg = np.delete(uv_ids, compare_list, axis=0)

        # FIXME this does not work if lifted_weights_all are none!
        # Extract the sub graph lifted mc problem
        uv_mask = np.in1d(uv_ids_lifted, seg_ids)
        uv_mask = uv_mask.reshape(uv_ids_lifted.shape).all(axis = 1)
        #uv_mask = np.swapaxes(np.reshape(uv_mask, uv_ids_lifted.shape), 0, 1)
        #uv_mask = np.logical_and(uv_mask[0], uv_mask[1])
        lifted_weights = lifted_weights_all[uv_mask]

        ids_in_mask = list(itertools.compress(xrange(len(uv_mask)), np.logical_not(uv_mask)))
        uv_ids_lifted_in_seg = np.delete(uv_ids_lifted, ids_in_mask, axis=0)

        # Now map the uv ids to locally consecutive ids
        # local graph (consecutive in obj)
        seg_ids_local, _, mapping = vigra.analysis.relabelConsecutive(seg_ids, start_label=0, keep_zeros=False)

        # mapping = old to new,
        # reverse = new to old
        reverse_mapping = {val: key for key, val in mapping.iteritems()}
        # edge dict
        uv_local = np.array([[mapping[u] for u in uv] for uv in uv_ids_in_seg])
        uv_local_lifted = np.array([[mapping[u] for u in uv] for uv in uv_ids_lifted_in_seg])

        # Next we want to introduce the lifted path edges
        if export_paths_path is None or not os.path.isfile(
                        export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id)):

            # Sample uv pairs out of seg_ids (make sure to have a minimal graph dist.)
            # ------------------------------------------------------------------------
            # TODO: Alternatively sample until enough false merges are found
            min_range = exp_params.min_nh_range
            max_sample_size = exp_params.max_sample_size
            uv_ids_paths_min_nh_local = compute_and_save_long_range_nh(
                uv_local,
                min_range,
                max_sample_size
            )

            if uv_ids_paths_min_nh_local.any():
                uv_ids_paths_min_nh_local = np.sort(uv_ids_paths_min_nh_local, axis = 1)

                # -------------------------------------------------------------
                # Get the distance transform of the current object

                masked_disttransf = deepcopy(disttransf)
                masked_disttransf[np.logical_not(mask)] = np.inf

                # Turn them to the original labels
                uv_ids_paths_min_nh = np.array([ np.array([reverse_mapping[u] for u in uv]) for uv in uv_ids_paths_min_nh_local])

                # Extract the respective coordinates from ecc_centers_seg thus creating pairs of coordinates
                uv_ids_paths_min_nh_coords = [[ecc_centers_seg[u] for u in uv] for uv in uv_ids_paths_min_nh]

                # Compute the shortest paths according to the pairs list
                paths_obj = shortest_paths(
                    masked_disttransf,
                    uv_ids_paths_min_nh_coords,
                    32) # TODO set n_threads from global params

            else:
                paths_obj = []

            # add the paths actually classified as being wrong if not already present
            extra_paths = false_paths[merge_id]
            # first we map them to segments
            extra_coords = [[tuple(p[0]), tuple(p[-1])] for p in extra_paths]
            extra_path_uvs = np.array([np.array(
                [mapping[seg[coord[0]]],
                 mapping[seg[coord[1]]]]) for coord in extra_coords])
            extra_path_uvs = np.sort(extra_path_uvs, axis=1)
            # Extra path uv pairs have to be different
            are_different = np.not_equal(extra_path_uvs[:, 0], extra_path_uvs[:, 1])
            extra_path_uvs = extra_path_uvs[are_different, :]
            extra_paths = extra_paths[are_different]

            if uv_ids_paths_min_nh_local.any():
                for extra_id, extra_uv in enumerate(extra_path_uvs):
                    if not any(np.equal(uv_ids_paths_min_nh_local, extra_uv).all(
                            1)):  # check whether this uv - pair is already present
                        paths_obj.append(extra_paths[extra_id])
                        uv_ids_paths_min_nh_local = np.append(uv_ids_paths_min_nh_local, extra_uv[None, :], axis=0)
            else:
                paths_obj = extra_paths.tolist()
                uv_ids_paths_min_nh_local = extra_path_uvs

        else:
            # Load paths
            print 'Loading lifted edges paths from: {}'.format(
                export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id)
            )
            with open(export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id), mode='r') as f:
                paths_obj = pickle.load(f)

            # We have to get the local uv ids of the loaded paths
            # This is a shorter version of the above
            extra_paths = paths_obj
            # first we map them to segments
            extra_coords = [[tuple(p[0]), tuple(p[-1])] for p in extra_paths]
            extra_path_uvs = np.array([np.array(
                [mapping[seg[coord[0]]],
                 mapping[seg[coord[1]]]]) for coord in extra_coords])
            extra_path_uvs = np.sort(extra_path_uvs, axis=1)
            uv_ids_paths_min_nh_local = np.empty((0, 2), extra_path_uvs.dtype)
            for extra_id, extra_uv in enumerate(extra_path_uvs):
                uv_ids_paths_min_nh_local = np.append(uv_ids_paths_min_nh_local, extra_uv[None, :], axis=0)

        if paths_obj:

            if export_paths_path is None or not os.path.isfile(export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id)):

                # Compute the path features
                features = path_feature_aggregator(ds, paths_obj, exp_params)
                features = np.nan_to_num(features)

                # Cache features for debug purpose
                with open(export_paths_path + '../debug/features_resolve_{}.pkl'.format(merge_id), mode='w') as f:
                    pickle.dump(features, f)

                # fs = path_feature_aggregator(ds, (extra_paths[0],), exp_params)
                # with open(export_paths_path + '../debug/fs_{}.pkl'.format(merge_id), mode='w') as f:
                #     pickle.dump(fs, f)

                # compute the lifted weights from rf probabilities
                lifted_path_weights = path_rf.predict_proba(features)[:,1]

                # Cache paths for evaluation purposes
                if export_paths_path is not None:
                    with open(export_paths_path + 'resolve_paths_{}.pkl'.format(merge_id), mode='w') as f:
                        pickle.dump(paths_obj, f)
                    with open(export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id), mode='w') as f:
                        pickle.dump(lifted_path_weights, f)

            else:
                # Load path probabilities
                print 'Loading lifted edges path weights from: {}'.format(
                    export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id)
                )
                with open(export_paths_path + 'resolve_paths_probs_{}.pkl'.format(merge_id), mode='r') as f:
                    lifted_path_weights = pickle.load(f)

            # Class 1: contain a merge
            # Class 0: don't contain a merge

            # scale the probabilities
            p_min = 0.001
            p_max = 1. - p_min
            lifted_path_weights = (p_max - p_min) * lifted_path_weights + p_min

            # Transform probs to weights
            lifted_path_weights = np.log((1 - lifted_path_weights) / lifted_path_weights)

            # Weighting edges with their length for proper lifted to local scaling
            lifted_path_weights /= lifted_path_weights.shape[0] * exp_params.lifted_path_weights_factor
            lifted_weights /= lifted_weights.shape[0]
            mc_weights /= mc_weights.shape[0]

            # Concatenate all lifted weights and edges
            # FIXME this does not work if lifted_weights_all are none!
            lifted_weights = np.concatenate(
                (lifted_path_weights, lifted_weights),
                axis=0 # TODO check for correct axis
            )
            uv_ids_lifted_nh_total = np.concatenate(
                (uv_ids_paths_min_nh_local, uv_local_lifted),
                axis=0 # TODO check for correct axis
            )

            resolved_nodes = optimizeLifted(
                uv_local,
                uv_ids_lifted_nh_total,
                mc_weights,
                lifted_weights
            )

            resolved_nodes, _, _ = vigra.analysis.relabelConsecutive(resolved_nodes, start_label = 0, keep_zeros = False)
            # project back to global node ids and save
            resolved_objs[merge_id] = {reverse_mapping[i] : node_res for i, node_res in enumerate(resolved_nodes)}

    return resolved_objs


def project_resolved_objects_to_segmentation(ds,
        seg_id,
        mc_segmentation,
        resolved_objs):

    rag = ds._rag(seg_id)
    mc_labeling, _ = rag.projectBaseGraphGt( mc_segmentation )
    new_label_offset = np.max(mc_labeling) + 1
    for obj in resolved_objs:
        resolved_nodes = resolved_objs[obj]
        for node_id in resolved_nodes:
            mc_labeling[node_id] = new_label_offset + resolved_nodes[node_id]
        new_label_offset += np.max(resolved_nodes.values()) + 1
    return rag.projectLabelsToBaseGraph(mc_labeling)


# FIXME why ?! this is just code copy from the function that trains the rf
def pre_compute_paths(
        data_sets,
        mc_segs,
        mc_segs_keys,
        params,
        paths_save_folder
):

    # loop over the training datasets
    for ds_id, paths_to_betas in enumerate(mc_segs):

        all_paths = []
        all_paths_to_objs = []
        all_path_classes = []

        paths_save_path = None if paths_save_folder == None else os.path.join(
            paths_save_folder,
            'path_%s.pkl' % '_'.join([data_sets[ds_id].ds_name])
        )
        cached_paths = []
        print "Looking for paths folder: {}".format(paths_save_path)

        current_ds = data_sets[ds_id]
        keys_to_betas = mc_segs_keys[ds_id]
        assert len(keys_to_betas) == len(paths_to_betas), "%i, %i" % (len(keys_to_betas), len(paths_to_betas))

        # Load ground truth
        gt = current_ds.gt()
        # we need this to safely replace this with the actual distance transforms later
        current_ds.add_input_from_data(np.zeros_like(gt.shape))

        # Initialize correspondence list which makes sure that the same merge is not extracted from
        # multiple mc segmentations
        if params.paths_avoid_duplicates:
            correspondence_list = []
        else:
            correspondence_list = None

        # loop over the different beta segmentations per train set
        for seg_id, seg_path in enumerate(paths_to_betas):
            key = keys_to_betas[seg_id]

            # Calculate the new distance transform and replace it in the dataset inputs
            seg = vigra.readHDF5(seg_path, key)
            seg = remove_small_segments(seg)
            dt  = distance_transform(seg, [1.,1.,params.anisotropy_factor])
            # NOTE IMPORTANT: We assume that the distance transform always has the last inp_id and that a (dummy) dt was already added in the beginning
            ds.replace_inp_from_data(ds.n_inp - 1, dt, clear_cache = False)
            # we delete all filters based on the distance transform
            ds.clear_filters(ds.n_inp - 1)

            # Compute the paths
            paths, paths_to_objs, path_classes, correspondence_list = extract_paths_and_labels_from_segmentation(
                current_ds,
                seg,
                params,
                gt,
                correspondence_list)

            all_paths.append(paths)
            all_paths_to_objs.append(paths_to_objs)
            all_path_classes.append(path_classes)

            if paths:
                pass

            else:
                print "No paths found for seg_id = {}".format(seg_id)

        print "Saving paths to:", paths_save_path
        with open(paths_save_path, 'w') as f:
            pickle.dump(
                {
                    'paths': all_paths,
                    'paths_to_objs': all_paths_to_objs,
                    'path_classes': all_path_classes
                }, f
            )
