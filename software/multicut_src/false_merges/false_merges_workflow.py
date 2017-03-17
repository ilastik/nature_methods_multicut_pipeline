
from remove_small_objects import RemoveSmallObjectsParams, remove_small_objects
from compute_paths_and_features import shortest_paths
from multicut_src import probs_to_energies
from lifted_mc import compute_and_save_lifted_nh
from lifted_mc import compute_and_save_long_range_nh
from multicut_src import learn_and_predict_rf_from_gt
# from find_false_merges_src import path_features_from_feature_images
# from find_false_merges_src import path_classification
from false_merges import path_feature_aggregator
from compute_paths_and_features import FeatureImageParams
from multicut_src.Tools import cache_name
from compute_border_contacts import compute_path_end_pairs, compute_border_contacts

import numpy as np
import vigra
import os


class ComputeFalseMergesParams:

    def __init__(
            self,
            remove_small_objects=RemoveSmallObjectsParams(),
            feature_images=FeatureImageParams(),
            paths_penalty_power=10,
            anisotropy_factor=10
    ):

        self.remove_small_objects = remove_small_objects
        self.feature_images=feature_images
        self.anisotropy_factor=anisotropy_factor
        self.paths_penalty_power=paths_penalty_power


def accumulate_paths_and_features():
    pass


# TODO move all training related stuff here
# cache the random forest here
def train_random_forest_for_merges(
        trainsets, # list of datasets with training data
        mc_segs_train, # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        gtruths,
        gtruths_keys,
        params
):

    import shutil
    from copy import deepcopy

    features_train = []
    labels_train = []
    # loop over the training datasets
    for ds_id, paths_to_betas in enumerate(mc_segs_train):
        current_ds = trainsets[ds_id]
        keys_to_betas = mc_segs_train_keys[ds_id]
        assert len(keys_to_betas) == len(paths_to_betas)

        # TODO: Add gt to dataset
        # Load ground truth
        gt_file = gtruths[ds_id]
        gt_key = gtruths_keys[ds_id]
        gt = vigra.readHDF5(gt_file, gt_key)

        # Initialize correspondence list which makes sure that the same merge is not extracted from
        # multiple mc segmentations
        correspondence_list = []

        # loop over the different beta segmentations per train set
        for seg_id, seg_path in enumerate(paths_to_betas):

            # TODO: Put the inside of this loop into accumulate_paths_and_features()
            """ INPUTS NEEDED IN THIS LOOP

            seg_path: path to mc segmentation (-> beta...)
            key: internal key to mc segmentation
            params: some parameter class (still needs implementation)
            current_ds: the current dataset of the sample (A_0, ...)

            """

            # load the segmentation
            key = keys_to_betas[seg_id]
            seg = vigra.readHDF5(seg_path, key)
            # TODO refactor params, parallelize internally if this becomes bottleneck
            seg = remove_small_objects(seg, params=params.remove_small_objects)

            # Delete distance transform and filters from cache
            # Generate file name according to how the cacher generated it (append parameters)
            # Find and delete the file if it is there
            dt_args = (current_ds, 0, [1., 1., params.anisotropy_factor])
            filepath = cache_name('distance_transform', 'dset_folder', True, False, *dt_args)
            if os.path.isfile(filepath):
                os.remove(filepath)

            # Clear filter cache
            filters_filepath = current_ds.cache_folder + '/filters/filters_10/distance_transform'
            if os.path.isdir(filters_filepath):
                shutil.rmtree(filters_filepath)

            # Compute distance transform on beta
            # FIXME: It would be nicer with keyword arguments (the cacher doesn't accept them)
            dt = current_ds.distance_transform(seg, *dt_args[1:])

            # Compute path end pairs
            border_contacts = compute_border_contacts(seg, dt)
            # This is supposed to only return those pairs that will be used for path computation
            # TODO: Throw out some under certain conditions (see also within function)
            path_pairs, paths_to_objs, path_classes, path_gt_labels, correspondence_list = compute_path_end_pairs(
                border_contacts, gt, correspondence_list, params
            )

            # TODO: Compute paths , TODO parallelize, internally
            all_paths = []

            # Invert the distance transform
            dt = np.amax(dt) - dt
            # Penalty power on distance transform
            dt = np.power(dt, 10)

            for obj in np.unique(paths_to_objs):
                # TODO implement shortest paths with labels
                # TODO clean paths for duplicate paths in this function

                # # Get the distance transform with correct penalty_power
                # dt = current_ds.distance_transform(
                #     seg, params.paths_penalty_power, [1., 1., params.anisotropy_factor])

                # Mask distance transform to current object
                masked_dt = deepcopy(dt)
                masked_dt[seg != obj] = np.inf

                # Take only the relevant path pairs
                pairs_in = np.array(path_pairs)[np.where(np.array(paths_to_objs) == obj)[0]]

                paths = shortest_paths(masked_dt, pairs_in)
                # paths is now a list of numpy arrays
                all_paths.extend(paths)

            # TODO: Here we have to ensure that every path is actually computed
            # TODO:  --> Throw not computed paths out of the lists

            # TODO: Remove paths under certain criteria
            # TODO: Do this only if GT is supplied
            # a) Class 'non-merged': Paths cross labels in GT multiple times
            # b) Class 'merged': Paths have to contain a certain amount of pixels in both GT classes

            # TODO: Extract features from paths
            # TODO: decide which filters and sigmas to use here (needs to be exposed first)
            features_train.append(
                path_feature_aggregator(current_ds, all_paths, params.anisotropy_factor)
            )
            labels_train.append(path_classes)

        features_train = np.concatenate(features_train, axis=0)  # TODO correct axis ?
        labels_train = np.concatenate(labels_train, axis=0)  # TODO correct axis ?

    return []


# TODO predict for test dataset
def predict_false_merge_paths(rf, mc_seg_test, mc_seg_test_key, params):

    # TODO load all test stuff
    seg_test = vigra.readHDF5(mc_seg_test, mc_seg_test_key)
    seg_test = remove_small_objects(
        image=seg_test, params=params.remove_small_objects
    )

    return []

"""
compute_false_merges(...):
    rf = train_random_forest_for_merges(...)
    false_merges = predict_false_merge_paths(rf, ...)
"""

def compute_false_merges(
        trainsets, # list of datasets with training data
        ds_test, # one dataset -> predict the false merged objects
        mc_segs_train, # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        mc_seg_test,
        mc_seg_test_key,
        gtruths,
        gtruths_keys,
        params=ComputeFalseMergesParams()
):
    """
    Computes and returns false merge candidates

    :param ds_train: Array of datasets representing multiple source images; [N x 1]
        Has to contain:
        ds_train.inp(0) := raw image
        ds_train.inp(1) := probs image

    :param ds_test:
        Has to contain:
        ds_train.inp(0) := raw image
        ds_train.inp(1) := probs image

    :param mc_segs_train: Multiple multicut segmentations on ds_train
        Different betas for each ds_train; [N x len(betas)]
    :param mc_seg_test: Multicut segmentation on ds_test (usually beta=0.5)
    :param params:
    :return:
    """

    assert len(trainsets) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"
    assert len(mc_segs_train_keys) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"

    # TODO: Store results of each step???
    # TODO: What are the images I can extract from ds_*???

    # The pipeline
    # ------------

    rf = train_random_forest_for_merges(
        trainsets,
        mc_segs_train,
        mc_segs_train_keys,
        gtruths,
        gtruths_keys,
        params
    )

    # TODO: Do the same things for the test data
    false_merges = predict_false_merge_paths(rf, mc_seg_test, mc_seg_test_key, params)

    # TODO: Random forest classification
    # Train on the betas
    # Get merge candidates on the test data

    # TODO: Return the labels of potential merges and associated paths
    return [], []


def resolve_merges_with_lifted_edges(
        ds, false_merge_ids, false_paths, path_classifier,
        feature_images, mc_segmentation, edge_probs,
        exp_params, pf_params, path_params, rf_params
):

    """

    seg_id = 0
    # resolve for each object individually
    for merge_id in false_merge_ids:
        # get the over-segmentatin and get fragmets corresponding to merge_id
        seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume
        mask = mc_segmentation == merge_id
        seg_ids = np.unique(seg[mask])
        # get the region adjacency graph
        rag = ds._rag(seg_id)
        # get the multicut weights
        uv_ids = rag.uvIds()
        # DONT IMPLEMENT THIS WAY
        edge_ids = []
        for e_id, u, v in enumerate(uv_ids):
            if u in seg_ids and v in seg_ids:
                edge_ids.append(e_id)
        # TODO beware of sorting
        mc_weights = probs_to_weights(ds, seg_id)
        mc_weights = mc_weights[edge_ids]

        # now we extracted the sub-graph multicut problem
        # next we want to introduce the lifted edges
        # sample uv pairs out of seg_ids (make sure to have a minimal graph dist.)

        # compute path features for the pairs (implemented)
        # classify the pathes (implemented)
        # transform probs to weights
        # add lifted_edges and solve lmc

    """

    seg_id = 0

    # Caluculate and cache the feature images if they are not cached
    # This is parallelized
    for _, feature_image in feature_images.iteritems():
        feature_image.compute_children(path_to_parent='', parallelize=True)

    disttransf = feature_images['segmentation'].get_feature('disttransf')
    # Pre-processing of the distance transform
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path
    #    detection) should be at the center of the current process
    disttransf = np.amax(disttransf) - disttransf
    #
    # c) Increase the value difference between pixels near the boundaries and pixels
    #    central within the processes. This increases the likelihood of the paths to
    #    follow the center of processes, thus avoiding short-cuts
    disttransf = np.power(disttransf, path_params.penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume
    print 'Computing eccentricity centers...'
    ecc_centers_seg = vigra.filters.eccentricityCenters(seg)
    ecc_centers_seg_dict = dict(zip(np.unique(seg), ecc_centers_seg))
    print ' ... done computing eccentricity centers.'

    # get the region adjacency graph
    rag = ds._rag(seg_id)

    mc_weights_all = probs_to_energies(ds, edge_probs, seg_id, exp_params)
    # mc_weights = mc_weights_all[edge_ids]

    # get the multicut weights
    uv_ids = rag.uvIds()

    for merge_id in false_merge_ids:

        mask = mc_segmentation == merge_id
        seg_ids = np.unique(seg[mask])

        compare = np.in1d(uv_ids, seg_ids)
        compare = np.swapaxes(np.reshape(compare, uv_ids.shape), 0, 1)
        compare = np.logical_and(compare[0], compare[1])
        mc_weights = mc_weights_all[compare]

        # ... now we extracted the sub-graph multicut problem!
        # Next we want to introduce the lifted edges

        # Sample uv pairs out of seg_ids (make sure to have a minimal graph dist.)
        # ------------------------------------------------------------------------
        import itertools
        compare_list = list(itertools.compress(xrange(len(compare)), np.logical_not(compare)))
        uv_ids_in_seg = np.delete(uv_ids, compare_list, axis=0)

        # local graph (consecutive in obj)
        seg_ids_local, _, mapping = vigra.analysis.relabelConsecutive(seg_ids, start_label=0)
        # mapping = old to new,
        # reverse = new to old
        reverse_mapping = {val: key for key, val in mapping.iteritems()}
        # edge dict
        uv_local = np.array([[mapping[u] for u in uv] for uv in uv_ids_in_seg])

        # TODO: This as parameter
        # TODO: Move sampling here
        # TODO: Sample until enough false merges are found
        min_range = 3
        max_sample_size = 10
        uv_ids_lifted_min_nh_local, all_uv_ids = compute_and_save_long_range_nh(
            uv_local, min_range, max_sample_size=max_sample_size, return_non_sampled=True
        )

        # TODO: Compute the paths from the centers of mass of the pairs list
        # -------------------------------------------------------------
        # Get the distance transform of the current object

        import copy
        masked_disttransf = copy.deepcopy(disttransf)
        masked_disttransf[np.logical_not(mask)] = np.inf

        # Turn them to the original labels
        uv_ids_lifted_min_nh = np.array([[reverse_mapping[u] for u in uv] for uv in uv_ids_lifted_min_nh_local])

        # Extract the respective coordinates from ecc_centers_seg thus creating pairs of coordinates
        uv_ids_lifted_min_nh_coords = [[ecc_centers_seg_dict[u] for u in uv] for uv in uv_ids_lifted_min_nh]

        # Compute the shortest paths according to the pairs list
        bounds=None
        logger=None
        yield_in_bounds=False
        return_pathim=False
        ps_computed = shortest_paths(
            masked_disttransf, uv_ids_lifted_min_nh_coords, bounds=bounds, logger=logger,
            return_pathim=return_pathim, yield_in_bounds=yield_in_bounds
        )

        # Compute the path features
        # TODO: Cache the path features?
        features = path_features_from_feature_images(ps_computed, feature_images, pf_params)

        # # classify the paths (implemented)
        # I will need:
        #   - The random forest
        #   - The features
        path_probs = path_classification(features, rf_params)

        # Class 0: 'false paths', i.e. containing a merge
        # Class 1: 'true paths' , i.e. not containing a merge

        # TODO: Do this:
        # # This is from probs_to_energies():
        # # ---------------------------------
        #
        # # scale the probabilities
        # # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
        # p_min = 0.001
        # p_max = 1. - p_min
        # edge_probs = (p_max - p_min) * edge_probs + p_min

        # Transform probs to weights
        # TODO: proper function?
        lifted_weights = np.log((1 - path_probs[:, 0]) / path_probs[:, 0])


        #
        # # probabilities to energies, second term is boundary bias
        # edge_energies = np.log((1. - edge_probs) / edge_probs) + np.log(
        #     (1. - exp_params.beta_local) / exp_params.beta_local)

        # # add lifted_edges and solve lmc
        # I will need:
        #   - mc_weights, edges
        #   - lifted_weights, edges

        # run_mc_solver(n_var, uv_ids, edge_energies, mc_params)


        # TODO : DEBUGGING!!!
        # TODO: Debug images
        # TODO: Look at paths

