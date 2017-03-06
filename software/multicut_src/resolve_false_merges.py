
from detect_false_merges import RemoveSmallObjectsParams, pipeline_remove_small_objects
from find_false_merges_src import shortest_paths
from multicut_src import probs_to_energies
from lifted_mc import compute_and_save_lifted_nh
from multicut_src import learn_and_predict_rf_from_gt

import numpy as np
import vigra


class PipelineParameters:

    def __init__(self,
                 remove_small_objects=RemoveSmallObjectsParams()):

        self.remove_small_objects = remove_small_objects


def compute_false_merges(
        ds_train, ds_test,
        mc_seg_train, mc_seg_test,
        params
):

    # TODO: Store results of each step???
    # TODO: What are the images I can extract from ds_*???

    # The pipeline
    # ------------

    # 1. Remove small objects
    mc_seg_train = pipeline_remove_small_objects(
        image=mc_seg_train, params=params.remove_small_objects
    )
    mc_seg_test = pipeline_remove_small_objects(
        image=mc_seg_test, params=params.remove_small_objects
    )

    # 2. Calculate feature images (including distance transform)
    # TODO: How do I implement the different betas???
    # compute_feature_images([], [], mc_seg_test)

    # 3. Compute border contacts

    # 4. Compute paths

    # 5. Extract features from paths

    # 6. Random forest classification

    # Return the labels of potential merges and associated paths
    return [], []


def resolve_merges_with_lifted_edges(
        ds, false_merge_ids, false_paths, path_classifier,
        feature_images, seg_id_in_feature_images, mc_segmentation, edge_probs,
        exp_params
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

    # TODO: Things I will need
    # TODO: 1. ds including oversegmentation, raw data, and probabilities
    # TODO: 2. false_merge_ids and respective false_paths
    # TODO: 3. Random forest classifier for paths to classify intermediate paths
    # TODO: 4. multi cut segmentation

    # TODO: use learn_and_predict_rf_from_gt for edge_probs
    seg_id = 0

    for merge_id in false_merge_ids:

        # get the over-segmentatin and get fragmets corresponding to merge_id
        seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume
        mask = mc_segmentation == merge_id
        seg_ids = np.unique(seg[mask])
        # get the region adjacency graph
        rag = ds._rag(seg_id)

        # get the multicut weights
        uv_ids = rag.uvIds()

        # # DONT IMPLEMENT THIS WAY
        # edge_ids = []
        # for e_id, uv in enumerate(uv_ids):
        #     if uv[0] in seg_ids and uv[1] in seg_ids:
        #         edge_ids.append(e_id)
        # # This is used for probs extraction of
        # #   edge_probs
        #
        # # # TODO beware of sorting
        # # edge_probs = learn_and_predict_rf_from_gt
        #

        mc_weights_all = probs_to_energies(ds, edge_probs, seg_id, exp_params)
        # mc_weights = mc_weights_all[edge_ids]

        compare = np.in1d(uv_ids, seg_ids)
        compare = np.swapaxes(np.reshape(compare, uv_ids.shape), 0, 1)
        compare = np.logical_and(compare[0], compare[1])
        # compare2 = np.swapaxes(np.array((compare, compare)), 0, 1)
        mc_weights = mc_weights_all[compare]

        # ... now we extracted the sub-graph multicut problem!
        # Next we want to introduce the lifted edges

        # # sample uv pairs out of seg_ids (make sure to have a minimal graph dist.)
        print 'Computing uv pairs for lifted nh ...'
        # This produces all possible combinations

        # ... but no minimum distance

        # # TODO: Exclude all pairs which are computed by compute_and_save_lifted_nh
        # # Use uv_ids[compare] as input
        #
        # # uv_ids_in_seg = uv_ids[compare2]
        # import itertools
        # compare_list = list(itertools.compress(xrange(len(compare)), np.logical_not(compare)))
        # uv_ids_in_seg = np.delete(uv_ids, compare_list, axis=0)
        #
        # # local graph (consecutive in obj)
        # seg_ids_local, _, mapping = vigra.analysis.relabelConsecutive(seg_ids, start_label=0, keep_zeros = False)
        # # mapping = old to new,
        # # reverse = new to old
        # reverse_mapping = {val : key for key, val in mapping}
        # # edge dict
        # #...
        # #uv_local = [uv_ids_in_seg[]]
        #
        # min_range = 3
        # uv_ids_lifted_min_nh = compute_and_save_long_range_nh(uv_local, min_range)
        uv_ids_lifted_min_nh = []

        # TODO: Playground
        feature_images[0].compute_children(path_to_parent='', n_threads=3)

        # TODO: Compute the paths from the centers of mass of the pairs list
        # -------------------------------------------------------------
        # First get the distance transform of the current object
        disttransf = feature_images[seg_id_in_feature_images].get_feature('disttransf')
        disttransf[np.logical_not(mask)] = 0

        # Compute the shortest paths according to the pairs list
        bounds=None
        logger=None
        yield_in_bounds=False
        return_pathim=False
        ps_computed = shortest_paths(
            disttransf, uv_ids_lifted_min_nh, bounds=bounds, logger=logger,
            return_pathim=return_pathim, yield_in_bounds=yield_in_bounds
        )

        # TODO: Compute path features for the pairs (implemented)
        # -------------------------------------------------
        # Load the feature images from cache or calculate them
        feature_image = feature_images['Somepath']

        # TODO: Do some parallelization here

        # TODO: extract the region features along the paths
        # TODO: Create some working image with a path in it
        path_image = np.array()
        # TODO: For each feature image extract the region features
        featurelist = ['Sum', 'Skewness', '...']
        vigra.analysis.extractRegionFeatures(
                            np.array(feature_image).astype(np.float32),
                            path_image, ignoreLabel=0,
                            features=featurelist
                        )

        pass
        # # classify the paths (implemented)

        # # transform probs to weights

        # # add lifted_edges and solve lmc




