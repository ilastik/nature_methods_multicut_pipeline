import vigra
import vigra.graphs as graphs
import numpy as np
from concurrent import futures

# from .. import DataSet
#from .. import cacher_hdf5

# class FeatureImageParams:
#
#     def __init__(self,
#                  filter_names=["gaussianSmoothing",
#                                "hessianOfGaussianEigenvalues",
#                                "laplacianOfGaussian"],
#                  sigmas=[1.6, 4.2, 8.3]
#                  ):
#         self.filter_names = filter_names
#         self.sigmas=sigmas


def shortest_paths(indicator,
        pairs,
        n_threads = 1):
    """
    This function was copied from processing_lib.py
    :param indicator:
    :return:
    """

    gridgr = graphs.gridGraph(indicator.shape)
    gridgr_edgeind = graphs.implicitMeanEdgeMap(gridgr, indicator.astype('float32'))

    def single_path(pair, instance = None):
        source = pair[0]
        target = pair[1]
        print 'Calculating path from {} to {}'.format(source, target)
        if instance == None:
            instance = graphs.ShortestPathPathDijkstra(gridgr)

        targetNode = gridgr.coordinateToNode(target)
        sourceNode = gridgr.coordinateToNode(source)

        instance.run(gridgr_edgeind, sourceNode, target=targetNode)
        path = instance.path(pathType='coordinates')
        if path.any():
            return path

    if n_threads > 1:
        print "Multi-threaded w/ n-threads = ", n_threads
        with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
            tasks = [executor.submit(single_path, pair) for pair in pairs]
            paths = [t.result() for t in tasks]
    else:
        print "Single threaded"
        instance = graphs.ShortestPathPathDijkstra(gridgr)
        paths = [single_path(pair, instance) for pair in pairs]

    return paths


# convenience function to combine path features
# TODO code different features with some keys
def path_feature_aggregator(ds, paths, params):
    # TODO move all params to exp_params
    # class Params:
    #     def __init__(self):
    #         #TODO use quantiles instead of Max and Min ?!
    #         self.stats = ["Mean","Variance","Sum","Maximum","Minimum","Kurtosis","Skewness"]
    #         self.max_threads = 8
    # params = Params()
    #
    anisotropy_factor = params.anisotropy_factor

    return np.concatenate([
        path_features_from_feature_images(ds, 0, paths, anisotropy_factor, params),
        path_features_from_feature_images(ds, 1, paths, anisotropy_factor, params),
        path_features_from_feature_images(ds, 'distance_transform', paths, anisotropy_factor, params),
        compute_path_lengths(paths, [1.,1.,anisotropy_factor]) ],
        axis = 1)


# TODO this could be parallelized over the paths
# compute the path lens for all paths
def compute_path_lengths(paths, anisotropy):
    """
    Computes the length of a path

    :param path:
        list( np.array([[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [xm1, xm2, ..., xmn]]) )
        with n dimensions and m coordinates
    :param anisotropy: [a1, a2, ..., an]
    :return: path lengtht list(float)
    """
    def compute_path_length(path, aniso_temp):
        #pathlen = 0.
        #for i in xrange(1, len(path)):
        #    add2pathlen = 0.
        #    for j in xrange(0, len(path[0, :])):
        #        add2pathlen += (anisotropy[j] * (path[i, j] - path[i - 1, j])) ** 2

        #    pathlen += add2pathlen ** (1. / 2)
        # TODO check that this actually agrees
        path_euclidean_diff = aniso_temp * np.diff(path, axis=0)
        path_euclidean_diff = np.sqrt(
                np.sum( np.square(path_euclidean_diff), axis=1 ) )
        return np.sum(path_euclidean_diff, axis=0)
    aniso_temp = np.array(anisotropy)
    return np.array([compute_path_length(np.array(path), aniso_temp) for path in paths])[:,None]


# don't cache for now
def path_features_from_feature_images(
        ds,
        inp_id,
        paths,
        anisotropy_factor,
        params):

    # FIXME for now we don't use fastfilters here
    feat_paths = ds.make_filters(inp_id, anisotropy_factor, use_fastfilters = False)
    #print feat_paths
    # TODO sort the feat_path correctly
    # load the feature images ->
    # FIXME this might be too memory hungry if we have a large global bounding box

    # compute the global bounding box
    global_min = np.min(
            np.concatenate([np.min(path, axis = 0)[None,:] for path in paths], axis=0),
            axis = 0
            )
    global_max = np.max(
            np.concatenate([np.max(path, axis = 0)[None,:] for path in paths], axis=0),
            axis = 0
            ) + 1
    # substract min coords from all paths to bring them to new coordinates
    paths_in_roi = [path - global_min for path in paths]
    roi = np.s_[global_min[0]:global_max[0],
            global_min[1]:global_max[1],
            global_min[2]:global_max[2]]

    # load features in global boundng box
    feature_volumes = []
    import h5py
    for path in feat_paths:
        with h5py.File(path) as f:
            feat_shape = f['data'].shape
            # we add a singleton dimension to single channel features to loop over channel later
            if len(feat_shape) == 3:
                feature_volumes.append(f['data'][roi][...,None])
            else:
                feature_volumes.append(f['data'][roi])
    stats = params.feature_stats

    def extract_features_for_path(path):

        # calculate the local path bounding box
        min_coords  = np.min(path, axis = 0)
        max_coords = np.max(path, axis = 0)
        max_coords += 1
        shape = tuple( max_coords - min_coords)
        path_image = np.zeros(shape, dtype='uint32')
        path -= min_coords
        # we swapaxes to properly index the image properly
        path_sa = np.swapaxes(path, 0, 1)
        path_image[path_sa[0], path_sa[1], path_sa[2]] = 1

        path_features = []
        for feature_volume in feature_volumes:
            for c in range(feature_volume.shape[-1]):
                path_roi = np.s_[min_coords[0]:max_coords[0],
                        min_coords[1]:max_coords[1],
                        min_coords[2]:max_coords[2],
                        c]
                extractor = vigra.analysis.extractRegionFeatures(
                        feature_volume[path_roi],
                        path_image,
                        ignoreLabel = 0,
                        features = stats)
                path_features.extend( [extractor[stat][1] for stat in stats]) # TODO make sure that dimensions match for more that 1d stats!
        #ret = np.array(path_features)[:,None]
        #print ret.shape
        return np.array(path_features)[None,:]

    # We parallelize over the paths for now.
    # TODO parallelizing over filters might in fact be much faster, because
    # we avoid the single threaded i/o in the beginning!
    # it also lessens memory requirements if we have less threads than filters
    # parallel
    with futures.ThreadPoolExecutor(max_workers = params.n_threads) as executor:
        tasks = []
        for p_id, path in enumerate(paths_in_roi):
            tasks.append( executor.submit( extract_features_for_path, path) )
        out = np.concatenate([t.result() for t in tasks], axis = 0)

    # serial for debugging
    #out = []
    #for p_id, path in enumerate(paths_in_roi):
    #    out.append( extract_features_for_path(path) )
    #out = np.concatenate(out, axis = 0)

    assert out.ndim == 2, str(out.shape)
    assert out.shape[0] == len(paths), str(out.shape)
    # TODO checkfor correct number of features

    return out

