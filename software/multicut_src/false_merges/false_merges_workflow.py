from compute_paths_and_features import shortest_paths
from multicut_src import probs_to_energies
from multicut_src import remove_small_segments
from multicut_src import compute_and_save_long_range_nh
from multicut_src import learn_and_predict_rf_from_gt
# from find_false_merges_src import path_features_from_feature_images
# from find_false_merges_src import path_classification
from compute_paths_and_features import FeatureImageParams, path_feature_aggregator
from multicut_src.Tools import cache_name
from compute_border_contacts import compute_path_end_pairs, compute_path_end_pairs_and_labels, compute_border_contacts

import numpy as np
import vigra
import os
import cPickle as pickle
import shutil
from copy import deepcopy


# TODO refactor params
class ComputeFalseMergesParams:

    def __init__(
            self,
            feature_images=FeatureImageParams(),
            paths_penalty_power=10,
            anisotropy_factor=10
    ):

        self.feature_images=feature_images
        self.anisotropy_factor=anisotropy_factor
        self.paths_penalty_power=paths_penalty_power

def extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        params):

    seg = vigra.readHDF5(seg_path, key)
    # FIXME we don't remove small objects for now, because this would relabel the segmentation, which we don't want in this case

    # Compute distance transform on beta
    dt = ds.distance_transform(seg, [1., 1., params.anisotropy_factor])
    # Compute path end pairs
    # TODO parallelize this function !
    border_contacts = compute_border_contacts(seg, dt)
    path_pairs, paths_to_objs = compute_path_end_pairs(border_contacts)

    # Invert the distance transform
    dt = np.amax(dt) - dt
    # Penalty power on distance transform
    dt = np.power(dt, 10)

    # compute the actual paths
    # TODO implement shortest paths with labels
    # TODO clean paths for duplicate paths in this function
    all_paths = shortest_paths(dt, path_pairs, n_threads = 20)

    return all_paths, paths_to_objs


# TODO refactor params
def extract_paths_and_labels_from_segmentation(
        ds,
        seg_path,
        key,
        params,
        gt,
        correspondence_list):
    """
    params:
    """

    # load the segmentation
    seg = vigra.readHDF5(seg_path, key)
    assert seg.shape == gt.shape
    seg = remove_small_segments(seg)

    # Compute distance transform on beta
    dt = ds.distance_transform(seg, [1., 1., params.anisotropy_factor])

    # Compute path end pairs
    # TODO parallelize this function !
    border_contacts = compute_border_contacts(seg, dt)
    # This is supposed to only return those pairs that will be used for path computation
    # TODO: Throw out some under certain conditions (see also within function)
    path_pairs, paths_to_objs, path_classes, path_gt_labels, correspondence_list = compute_path_end_pairs_and_labels(
        border_contacts, gt, correspondence_list
    )

    # Invert the distance transform
    dt = np.amax(dt) - dt
    # Penalty power on distance transform
    dt = np.power(dt, 10)

    # compute the actual paths
    # TODO implement shortest paths with labels
    # TODO clean paths for duplicate paths in this function
    all_paths = shortest_paths(dt, path_pairs, n_threads = 20)

    # TODO FIXME as far as I can see, we don't need this loop, it does not brinng anything,
    # but makes the computations inefficient as hell...
    #all_paths = []
    #for obj in np.unique(paths_to_objs):

    #    # Mask distance transform to current object
    #    masked_dt = deepcopy(dt)
    #    masked_dt[seg != obj] = np.inf

    #    # Take only the relevant path pairs
    #    pairs_in = np.array(path_pairs)[np.where(np.array(paths_to_objs) == obj)[0]]

    #    paths = shortest_paths(masked_dt, pairs_in, n_threads = 1)
    #    # paths is now a list of numpy arrays
    #    all_paths.extend(paths)

    # TODO: Here we have to ensure that every path is actually computed
    # TODO:  --> Throw not computed paths out of the lists

    # TODO: Remove paths under certain criteria
    # TODO: Do this only if GT is supplied
    # a) Class 'non-merged': Paths cross labels in GT multiple times
    # b) Class 'merged': Paths have to contain a certain amount of pixels in both GT classes
    # TODO implement stuff here

    return all_paths, paths_to_objs, path_classes, correspondence_list


# TODO move all training related stuff here
# cache the random forest here
def train_random_forest_for_merges(
        trainsets, # list of datasets with training data
        mc_segs_train, # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        params,
        save_folder= None
):

    # TODO use vigra rf 3 once it is in the vigra conda package
    from sklearn.ensemble import RandomForestClassifier as Skrf

    caching = False
    if save_folder != None:
        caching = True
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    save_path = None if save_folder == None else os.path.join(
            save_folder,
            'rf_merges_%s.pkl' % '_'.join([ds.ds_name for ds in trainsets])
        ) # TODO more meaningful save name

    # check if the rf will be cached and if yes, if it is already cached
    if caching and os.path.exists(save_path):
        print "Loading rf from:", save_path
        with open(save_path) as f:
            rf = pickle.load(f)
    # otherwise do the actual calculations
    else:
        print save_path
        features_train = []
        labels_train = []
        # loop over the training datasets
        for ds_id, paths_to_betas in enumerate(mc_segs_train):
            current_ds = trainsets[ds_id]
            keys_to_betas = mc_segs_train_keys[ds_id]
            assert len(keys_to_betas) == len(paths_to_betas), "%i, %i" % (len(keys_to_betas), len(paths_to_betas))

            # Load ground truth
            gt = current_ds.gt()

            # Initialize correspondence list which makes sure that the same merge is not extracted from
            # multiple mc segmentations
            correspondence_list = []

            # loop over the different beta segmentations per train set
            for seg_id, seg_path in enumerate(paths_to_betas):
                key = keys_to_betas[seg_id]

                # Delete distance transform and filters from cache
                # Generate file name according to how the cacher generated it (append parameters)
                # Find and delete the file if it is there
                dt_args = (current_ds, [1., 1., params.anisotropy_factor])
                filepath = cache_name('distance_transform', 'dset_folder', True, False, *dt_args)
                if os.path.isfile(filepath):
                    os.remove(filepath)

                all_paths, paths_to_objs, path_classes, correspondence_list = extract_paths_and_labels_from_segmentation(
                        current_ds,
                        seg_path,
                        key,
                        params,
                        gt,
                        correspondence_list)

                # Clear filter cache
                filters_filepath = current_ds.cache_folder + '/filters/filters_10/distance_transform'
                if os.path.isdir(filters_filepath):
                    shutil.rmtree(filters_filepath)

                # TODO: Extract features from paths
                # TODO: decide which filters and sigmas to use here (needs to be exposed first)
                features_train.append(
                    path_feature_aggregator(current_ds, all_paths, params.anisotropy_factor)
                )
                labels_train.append(path_classes)

        features_train = np.concatenate(features_train, axis=0)  # TODO correct axis ?
        labels_train = np.concatenate(labels_train, axis=0)  # TODO correct axis ?
        assert features_train.shape[0] == labels_train.shape[0]

        # remove nans
        features_train = np.nan_to_num(features_train).astype('float32')

        # TODO vigra.rf3
        # TODO set n_threads from global param object
        n_threads = 8
        rf = Skrf(n_jobs = n_threads)
        rf.fit(features_train, labels_train)
        if caching:
            print "Saving path-rf to:", save_path
            with open(save_path, 'w') as f:
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

    rf = train_random_forest_for_merges(
        trainsets,
        mc_segs_train,
        mc_segs_train_keys,
        #gtruths,
        #gtruths_keys,
        params,
        rf_save_folder
    )

    paths_test, paths_to_objs_test = extract_paths_from_segmentation(
        ds_test,
        mc_seg_test,
        mc_seg_test_key,
        params)
    assert len(paths_test) == len(paths_to_objs_test)

    features_test = path_feature_aggregator(
            ds_test,
            paths_test,
            params.anisotropy_factor)
    assert features_test.shape[0] == len(paths_test)
    features_test = np.nan_to_num(features_test)
    # TODO vigra.rf3
    return paths_test, rf.predict_proba(features_test)[:,1], paths_to_objs_test # FIXME TODO do we keep first or second channel ?


def resolve_merges_with_lifted_edges(ds,
        seg_id,
        false_paths, # dict(merge_ids : false_paths)
        path_rf,
        mc_segmentation,
        mc_weights_all, # the precomputed mc-weights
        exp_params,
        penalty_power = 10 # TODO move this to exp_params
):
    assert isinstance(false_paths, dict)

    disttransf = ds.distance_trafo(mc_segmentation, [1.,1.,exp_params.anisotropy])
    # Pre-processing of the distance transform
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path
    #    detection) should be at the center of the current process
    disttransf = np.amax(disttransf) - disttransf
    #
    # c) Increase the value difference between pixels near the boundaries and pixels
    #    central within the processes. This increases the likelihood of the paths to
    #    follow the center of processes, thus avoiding short-cuts
    disttransf = np.power(disttransf, penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume
    print 'Computing eccentricity centers...'
    ecc_centers_seg = vigra.filters.eccentricityCenters(seg)
    ecc_centers_seg_dict = dict(zip(np.unique(seg), ecc_centers_seg))
    print ' ... done computing eccentricity centers.'

    # get the region adjacency graph
    rag = ds._rag(seg_id)

    # get the multicut weights
    uv_ids = rag.uvIds()

    for merge_id in false_paths:

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

        masked_disttransf = deepcopy(disttransf)
        masked_disttransf[np.logical_not(mask)] = np.inf

        # Turn them to the original labels
        uv_ids_lifted_min_nh = np.array([[reverse_mapping[u] for u in uv] for uv in uv_ids_lifted_min_nh_local])

        # Extract the respective coordinates from ecc_centers_seg thus creating pairs of coordinates
        uv_ids_lifted_min_nh_coords = [[ecc_centers_seg_dict[u] for u in uv] for uv in uv_ids_lifted_min_nh]

        # we initialize the false_paths with the paths actually classified as being wrong
        # TODO these need to be mapped to a lifted edge / added to the uv ids
        paths_obj = deepcopy(false_paths[merge_id])
        # Compute the shortest paths according to the pairs list
        paths_obj.extend( shortest_paths(
            masked_disttransf,
            uv_ids_lifted_min_nh_coords) )

        # Compute the path features
        # TODO: Cache the path features?
        features = path_feature_aggregator(ds, false_paths, exp_params.anisotropy_factor)
        path_probs = path_rf.predict_proba(features)[:,1] # TODO which channel ?

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

