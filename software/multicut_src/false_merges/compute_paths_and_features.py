import vigra
import vigra.graphs as graphs
import numpy as np
import scipy
from joblib import Parallel,delayed
from concurrent import futures
from time import time

from ..ExperimentSettings import ExperimentSettings
from ..MCSolverImpl import multicut_exact, weight_z_edges, weight_all_edges, weight_xyz_edges
from ..tools import find_matching_row_indices

from ..tools import cacher_hdf5

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty  # conda version build with cplex
    except ImportError:
        try:
            import nifty_with_gurobi as nifty  # conda version build with gurobi
        except ImportError:
            raise ImportError("No valid nifty version was found.")


# calculate the distance transform for the given segmentation
def distance_transform(segmentation, anisotropy):
    edge_volume = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmentation[z])[None, :]
         for z in xrange(segmentation.shape[0])],
        axis=0
    )
    dt = vigra.filters.distanceTransform(edge_volume.astype('uint32'), pixel_pitch=anisotropy, background=True)
    return dt


def extract_local_graph_from_segmentation(
        ds,
        seg_id,
        mc_segmentation,
        object_id,
        uv_ids,
        uv_ids_lifted=None
):

    seg = ds.seg(seg_id)
    mask = mc_segmentation == object_id
    seg_ids = np.unique(seg[mask])

    # map the extracted seg_ids to consecutive labels
    seg_ids_local, _, mapping = vigra.analysis.relabelConsecutive(
        seg_ids,
        start_label=0,
        keep_zeros=False
    )
    # mapping = old to new,
    # reverse = new to old
    reverse_mapping = {val: key for key, val in mapping.iteritems()}

    # mask the local uv ids in this object
    local_uv_mask = np.in1d(uv_ids, seg_ids)
    local_uv_mask = local_uv_mask.reshape(uv_ids.shape).all(axis=1)

    if uv_ids_lifted is not None:
        # mask the lifted uv ids in this object
        lifted_uv_mask = np.in1d(uv_ids_lifted, seg_ids)
        lifted_uv_mask = lifted_uv_mask.reshape(uv_ids_lifted.shape).all(axis=1)
        return local_uv_mask, lifted_uv_mask, mapping, reverse_mapping

    return local_uv_mask, mapping, reverse_mapping


# TODO take nifty shortest paths and parallelize
def shortest_paths(
        indicator,
        pairs,
        n_threads=1
):
    """
    This function was copied from processing_lib.py
    :param indicator:
    :return:
    """

    gridgr         = graphs.gridGraph(indicator.shape)
    gridgr_edgeind = graphs.implicitMeanEdgeMap(gridgr, indicator.astype('float32'))

    def single_path(pair, instance=None):
        source = pair[0]
        target = pair[1]
        print 'Calculating path from {} to {}'.format(source, target)
        if instance is None:
            instance = graphs.ShortestPathPathDijkstra(gridgr)

        targetNode = gridgr.coordinateToNode(target)
        sourceNode = gridgr.coordinateToNode(source)

        instance.run(gridgr_edgeind, sourceNode, target=targetNode)
        path = instance.path(pathType='coordinates')
        if path.any():
            return path

    if n_threads > 1:
        print "Multi-threaded w/ n-threads = ", n_threads
        with futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            tasks = [executor.submit(single_path, pair) for pair in pairs]
            paths = [t.result() for t in tasks]
    else:
        print "Single threaded"
        instance = graphs.ShortestPathPathDijkstra(gridgr)
        paths = [single_path(pair, instance) for pair in pairs]

    return paths


def path_feature_aggregator(ds, paths, mc_segmentation_name=None):
    anisotropy_factor = ExperimentSettings().anisotropy_factor
    return np.concatenate([
        path_features_from_feature_images(ds, 0, paths, anisotropy_factor, mc_segmentation_name),
        path_features_from_feature_images(ds, 1, paths, anisotropy_factor, mc_segmentation_name),
        # we assume that the distance transform is added as inp_id 2
        path_features_from_feature_images(ds, 2, paths, anisotropy_factor, mc_segmentation_name),
        compute_path_lengths(ds, paths, [anisotropy_factor, 1., 1.], mc_segmentation_name)],
        axis=1
    )


# convenience function to combine path features
# TODO code different features with some keys
# TODO include seg_id
def path_feature_aggregator_for_resolving(ds, paths, feature_volumes_0,
                            feature_volumes_1,
                            feature_volumes_2, mc_segmentation_name=None):
    return np.concatenate([
        path_features_from_feature_images_for_resolving(ds, paths, 0, feature_volumes_0, mc_segmentation_name),
        path_features_from_feature_images_for_resolving(ds, paths, 1, feature_volumes_1, mc_segmentation_name),
        # we assume that the distance transform is added as inp_id 2
        path_features_from_feature_images_for_resolving(ds, paths, 2, feature_volumes_2, mc_segmentation_name),
        compute_path_lengths(ds, paths, [ExperimentSettings().anisotropy_factor, 1., 1.],
                             mc_segmentation_name)],
        axis=1
    )



# TODO this could be parallelized over the paths
# compute the path lens for all paths
@cacher_hdf5("feature_folder", ignoreNumpyArrays=True)
def compute_path_lengths(ds, paths, anisotropy, append_to_cache_name):
    """
    Computes the length of a path

    :param path:
        list( np.array([[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [xm1, xm2, ..., xmn]]) )
        with n dimensions and m coordinates
    :param anisotropy: [a1, a2, ..., an]
    :return: path lengtht list(float)
    """
    def compute_path_length(path, aniso_temp):
        # pathlen = 0.
        # for i in xrange(1, len(path)):
        #    add2pathlen = 0.
        #    for j in xrange(0, len(path[0, :])):
        #        add2pathlen += (anisotropy[j] * (path[i, j] - path[i - 1, j])) ** 2

        #    pathlen += add2pathlen ** (1. / 2)
        # TODO check that this actually agrees
        path_euclidean_diff = aniso_temp * np.diff(path, axis=0)
        path_euclidean_diff = np.sqrt(np.sum(np.square(path_euclidean_diff), axis=1))
        return np.sum(path_euclidean_diff, axis=0)
    aniso_temp = np.array(anisotropy)
    return np.array([compute_path_length(np.array(path), aniso_temp) for path in paths])[:, None]

#out of the function for parallelizing
def python_region_features_extractor_2_mc(single_vals):

    path_features = []

    [path_features.extend([np.mean(vals), np.var(vals), sum(vals),
                            max(vals), min(vals),
                            scipy.stats.kurtosis(vals),
                           (((vals - vals.mean()) / vals.std(ddof=0)) ** 3).mean()
])
     for vals in single_vals]

    return np.array(path_features)

# don't cache for now
@cacher_hdf5("feature_folder", ignoreNumpyArrays=True)
def path_features_from_feature_images(
        ds,
        inp_id,
        paths,
        anisotropy_factor,
        append_to_cache_name
):

    # FIXME for now we don't use fastfilters here
    feat_paths = ds.make_filters(inp_id, anisotropy_factor)

    roi = np.s_[
        0:ds.shape[0],
        0:ds.shape[1],
        0:ds.shape[2]
    ]

    # load features in global boundng box
    feature_volumes = []
    import h5py
    print "loading h5py..."
    time_a=time()
    for path in feat_paths:
        with h5py.File(path) as f:
            feat_shape = f['data'].shape
            # we add a singleton dimension to single channel features to loop over channel later
            if len(feat_shape) == 3:
                feature_volumes.append(np.float64(f['data'][roi][..., None]))
            else:
                feature_volumes.append(np.float64(f['data'][roi]))
    time_b=time()
    print "loading h5py took ",time_b-time_a," secs"

    def python_region_features_extractor_sc(path):

        pixel_values = []

        for feature_volume in feature_volumes:
            for c in range(feature_volume.shape[-1]):
                pixel_values.extend([feature_volume[path[:, 0],
                                                    path[:, 1],
                                                    path[:, 2]]
                                                    [:, c]])

        return np.array(pixel_values)

    if len(paths) > 1:

        time1=time()
        pixel_values_all = [python_region_features_extractor_sc (path)
                              for idx,path in enumerate(paths)]
        time2 = time()
        print "pixel values took ",time2-time1," secs"
        out = np.array(Parallel(n_jobs=ExperimentSettings().n_threads) \
            (delayed(python_region_features_extractor_2_mc)(single_vals)
             for single_vals in pixel_values_all ))
        time3 = time()
        print "filters took ", time3 - time2, " secs"


    else:
        time1=time()
        pixel_values_all = [python_region_features_extractor_sc (path)
                              for idx,path in enumerate(paths)]
        time2 = time()
        print "pixel values took ",time2-time1," secs"
        out=np.array([python_region_features_extractor_2_mc(single_vals)
                            for single_vals in pixel_values_all])
        time3 = time()
        print "filters took ", time3 - time2, " secs"

    assert out.ndim == 2, str(out.shape)
    assert out.shape[0] == len(paths), str(out.shape)
    # TODO checkfor correct number of features

    return out

# don't cache for now
@cacher_hdf5("feature_folder", ignoreNumpyArrays=True)
def path_features_from_feature_images_for_resolving(
        ds,
        paths,
        ds_inp,
        feature_volumes,
        merge_id

):



    def python_region_features_extractor_sc(path):

        pixel_values = []

        for feature_volume in feature_volumes:
                pixel_values.extend([feature_volume[path[:, 0],
                                                    path[:, 1],
                                                    path[:, 2]]
                                                    ])

        return np.array(pixel_values)

    if len(paths) > 1:
    # if False:
        time1=time()
        pixel_values_all = [python_region_features_extractor_sc (path)
                              for idx,path in enumerate(paths)]
        time2 = time()
        print "pixel values took ",time2-time1," secs"
        out = np.array(Parallel(n_jobs=ExperimentSettings().n_threads) \
            (delayed(python_region_features_extractor_2_mc)(single_vals)
             for single_vals in pixel_values_all ))
        time3 = time()
        print "filters took ", time3 - time2, " secs"


    else:
        time1=time()
        pixel_values_all = [python_region_features_extractor_sc (path)
                              for idx,path in enumerate(paths)]
        time2 = time()
        print "pixel values took ",time2-time1," secs"
        out=np.array([python_region_features_extractor_2_mc(single_vals)
                            for single_vals in pixel_values_all])
        time3 = time()
        print "filters took ", time3 - time2, " secs"

    assert out.ndim == 2, str(out.shape)
    assert out.shape[0] == len(paths), str(out.shape)
    # TODO checkfor correct number of features

    return out


# features based on multicut in the object
# we calculate the multicut (TODO also include lifted) for each
# object and for different betas, project to the path edges and count the number of splits
def multicut_path_features(
        ds,
        seg_id,
        mc_segmentation,
        objs_to_paths,  # dict[merge_ids : dict[path_ids : paths]]
        edge_probabilities
):

    seg = ds.seg(seg_id)
    uv_ids = ds.uv_ids(seg_id)

    # find the local edge ids along the path
    def edges_along_path(path, mapping, uvs_local):
        edge_ids = []
        u = mapping[seg[path[0]]]  # TODO does this work ?
        for p in path[1:]:
            v = mapping[seg[p]]  # TODO does this work ?
            if u != v:
                uv = np.array([min(u, v), max(u, v)])
                edge_id = find_matching_row_indices(uvs_local, uv)[0, 0]
                edge_ids.append(edge_id)
            u = v
        return np.array(edge_ids)

    # needed for weight transformation
    weighting_scheme = ExperimentSettings.weighting_scheme
    weight           = ExperimentSettings.weight
    edge_areas       = ds.topology_features(seg_id, False)[:, 0].astype('uint32')
    edge_indications = ds.edge_indications(seg_id)

    # transform edge-probabilities to weights
    def to_weights(edge_mask, beta):

        probs = edge_probabilities[edge_mask]
        areas = edge_areas[edge_mask]
        indications = edge_indications[edge_mask]

        # scale the probabilities
        p_min = 0.001
        p_max = 1. - p_min
        probs = (p_max - p_min) * probs + p_min

        # probabilities to energies, second term is boundary bias
        weights = np.log((1. - probs) / probs) + np.log((1. - beta) / beta)

        # weight edges
        if weighting_scheme == "z":
            weights = weight_z_edges(weights, areas, indications, weight)
        elif weighting_scheme == "xyz":
            weights = weight_xyz_edges(weights, areas, indications, weight)
        elif weighting_scheme == "all":
            weights = weight_all_edges(weights, areas, weight)

        return weights

    # TODO more_feats ?!
    betas = np.arange(0.3, 0.75, 0.05)
    n_feats = len(betas)
    n_paths = np.sum([len(paths) for paths in objs_to_paths])
    features = np.zeros((n_paths, n_feats), dtype='float32')

    # TODO parallelize
    for obj_id in objs_to_paths:

        local_uv_mask, mapping, reverse_mapping = extract_local_graph_from_segmentation(
            mc_segmentation,
            obj_id,
            uv_ids
        )
        uv_local = np.array([[mapping[u] for u in uv] for uv in uv_ids[local_uv_mask]])
        n_var = uv_local.max() + 1

        path_edge_ids = {}
        for path_id, path in objs_to_paths[obj_id].iteritems():
            path_edge_ids[path_id] = edges_along_path(path, mapping, uv_local)

        for ii, beta in enumerate(betas):
            weights = to_weights(local_uv_mask, beta)
            node_labels, _, _ = multicut_exact(n_var, uv_local, weights)
            cut_edges = node_labels[[uv_local[:, 0]]] != node_labels[[uv_local[:, 1]]]

            for path_id, e_ids in path_edge_ids.iteritems():
                features[path_id, ii] = np.sum(cut_edges[e_ids])

    return features


def cut_watershed(graph, weights, source, sink):

    # TODO I don't know if this is the correct way to do this
    # make the seeds from source and sing
    seeds = np.zeros(graph.numberOfNodes, dtype='uint64')
    seeds[source] = 1
    seeds[sink] = 2
    node_labeling = nifty.graph.edgeWeightedWatershedsSegmentation(
        graph,
        seeds,
        weights
    )

    uvs = graph.uvIds()
    # return the edge labels (cut or not cut for each edge)
    return node_labeling[uvs[:, 0]] != node_labeling[uvs[:, 1]]


def cut_graphcut(graph, weights, source, sink):
    raise NotImplementedError("Cutting with graph cut is too cutting edge.")


def cut_seeded_agglomeration(graph, weights, source, sink):
    raise NotImplementedError("Cutting with seeded agglo not implemented yet.")


# features based on most likely cut along path (via graphcut)
# return edge features of corresponding cut, depending on feature list
def cut_features(
        ds,
        seg_id,
        mc_segmentation,
        objs_to_paths,  # dict[merge_ids : dict[path_ids : paths]]
        edge_weights,
        anisotropy_factor,
        feat_list=['raw', 'prob', 'distance_transform'],
        cut_method='watershed'
):

    cutters = {
        'watershed': cut_watershed,
        'graphcut': cut_graphcut,
        'seeded_clustering': cut_seeded_agglomeration
    }
    assert feat_list
    assert cut_method in cutters
    cutter = cutters[cut_method]

    seg = ds.seg(seg_id)
    uv_ids = ds.uv_ids(seg_id)

    # find the global and local edge ids along the path, as well as the local
    # start and ed point of the path
    def edges_along_path(path, mapping, uvs_local):

        edge_ids = []
        edge_ids_local = []

        u = seg[path[0]]  # TODO does this work ?
        u_local = mapping[seg[path[0]]]  # TODO does this work ?

        # find edge-ids along the path
        # if this turns out to be a bottleneck, we can c++ it
        for p in path[1:]:

            v = seg[p]  # TODO does this work ?
            v_local = mapping[seg[p]]  # TODO does this work ?

            if u != v:
                uv = np.array([min(u, v), max(u, v)])
                uv_local = np.array([min(u_local, v_local), max(u_local, v_local)])

                edge_id = find_matching_row_indices(uv_ids, uv)[0, 0]
                edge_id_local = find_matching_row_indices(uvs_local, uv_local)[0, 0]

                edge_ids.append(edge_id)
                edge_ids_local.append(edge_id_local)

            u = v
            u_local = v_local

        return np.array(edge_ids), np.array(edge_ids_local)

    # get the edge features already calculated for the mc-ppl
    edge_features = []
    if 'raw' in feat_list:
        edge_features.append(ds.edge_features(seg_id, 0, anisotropy_factor))
    if 'prob' in feat_list:
        edge_features.append(ds.edge_features(seg_id, 1, anisotropy_factor))
    if 'distance_transform' in feat_list:
        # we assume that the dt was already added as additional input
        edge_features.append(ds.edge_features(seg_id, 2, anisotropy_factor))
    edge_features = np.concatenate(edge_features, axis=1)

    n_paths = np.sum([len(paths) for paths in objs_to_paths])
    features = np.zeros((n_paths, edge_features.shape[1]), dtype='float32')

    # TODO parallelize
    for obj_id in objs_to_paths:

        local_uv_mask, mapping, reverse_mapping = extract_local_graph_from_segmentation(
            mc_segmentation,
            obj_id,
            uv_ids
        )
        uv_local = np.array([[mapping[u] for u in uv] for uv in uv_ids[local_uv_mask]])

        # local weights and graph for the cutter
        weights_local = edge_weights[local_uv_mask]
        graph_local = nifty.graph.UndirectedGraph(uv_local.max() + 1)
        graph_local.insertEdges(uv_local)

        # TODO run graphcut or edge based ws for each path, using end points as seeds
        # determine the cut edge and append to feats
        for path_id, path in objs_to_paths[obj_id].iteritems():
            edge_ids, edge_ids_local, source, sink = edges_along_path(path, mapping, uv_local)

            # get source and sink
            # == start and end node of the path
            source = mapping[seg[path[0]]]
            sink = mapping[seg[path[-1]]]

            # run cut with seeds at path end points
            # returns the cut edges
            local_two_coloring = cutter(graph_local, weights_local, source, sink)
            # find the cut edge along the path
            cut_edge = np.where(local_two_coloring == 1)[0]
            # make sure we only have 1 cut edge
            assert len(cut_edge) == 1
            cut_edge = cut_edge[0]

            # cut-edge project back to global edge-indexing and get according features
            global_edge = edge_ids[edge_ids_local == cut_edge]
            features[path_id] = edge_features[global_edge]

    return features
