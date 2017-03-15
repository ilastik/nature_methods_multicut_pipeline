import vigra
import vigra.graphs as graphs
import numpy as np
from concurrent import futures

# from .. import DataSet
#from .. import cacher_hdf5

class FeatureImageParams:

    def __init__(self,
                 filter_names=["gaussianSmoothing",
                               "hessianOfGaussianEigenvalues",
                               "laplacianOfGaussian"],
                 sigmas=[1.6, 4.2, 8.3]
                 ):
        self.filter_names = filter_names
        self.sigmas=sigmas


def shortest_paths(indicator,
        pairs,
        bounds=None,
        yield_in_bounds=False):
    """
    This function was copied from processing_lib.py
    :param indicator:
    :param pairs:
    :param bounds:
    :param yield_in_bounds:
    :return:
    """

    # Crate the grid graph and shortest path objects
    gridgr = graphs.gridGraph(indicator.shape)
    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    def compute_path_for_pair(pair):
        source = pair[0]
        target = pair[1]
        print 'Calculating path from {} to {}'.format(source, target)

        targetNode = gridgr.coordinateToNode(target)
        sourceNode = gridgr.coordinateToNode(source)

        instance.run(gridgr_edgeind, sourceNode, target=targetNode)
        path = instance.path(pathType='coordinates')
        if path.any():
            # Do not forget to correct for the offset caused by cropping!
            if bounds is not None:
                path = path + [bounds[0].start, bounds[1].start, bounds[2].start]
                if yield_in_bounds:
                    return path, paths_in_bounds
            else:
                return path

    # TODO this will not parallelize properly until the gil is lifted for ShortestPathPathDijkstra.run !
    n_threads = 1
    with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
        tasks = []
        for pair in pairs:
            tasks.append( executor.submit(compute_path_for_pair, pair) )

    if yield_in_bounds:
        results = [t.result() for t in tasks]
        paths = [res[0] for res in results]
        paths_in_bounds = [res[1] for res in results]
        return paths, paths_in_bounds
    else:
        return [t.result() for t in tasks]


# convenience function to combine path features
# TODO code different features with some keys
# TODO expose the filters and sigmas for experimentation
def path_feature_aggregator(ds, paths, anisotropy_factor):
    # TODO move all params to exp_params
    class Params:
        def __init__(self):
            self.stats = ["Mean","Variance"]
            self.max_threads = 8
    params = Params()
    #
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
    # TODO sort the feat_path correctly
    # load the feature images ->
    # FIXME this might be too memory hungry if we have a large global bounding box

    # compute the global bounding box
    min_coords = np.min(
            np.concatenate([np.min(path, axis = 0)[None,:] for path in paths], axis=0),
            axis = 0
            )
    max_coords = np.max(
            np.concatenate([np.max(path, axis = 0)[None,:] for path in paths], axis=0),
            axis = 0
            )
    max_coords += 1
    # substract min coords from all paths to bring them to new coordinates
    paths = [path - min_coords for path in paths]
    roi = np.s_[min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]]

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
    stats = params.stats

    def extract_features_for_path(path):

        # calculate the local path bounding box
        min_coors  = np.min(path, axis = 0)
        max_coords = np.max(path, axis = 0)
        max_coords += 1
        shape = tuple( max_coords - min_coords)
        path_image = np.zeros(shape, dtype='uint32')
        path -= min_coords
        # TODO FIXME why swap axes ????
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
                path_features.append( extractor[stat] for stat in stats) # TODO make sure that dimensions match for more that 1d stats!
        #ret = np.array(path_features)[:,None]
        #print ret.shape
        return np.array(path_features)[None,:]

    # We parallelize over the paths for now.
    # TODO parallelizing over filters might in fact be much faster, because
    # we avoid the single threaded i/o in the beginning!
    # it also lessens memory requirements if we have less threads than filters
    # parallel
    with futures.ThreadPoolExecutor(params.max_threads) as executor:
        tasks = []
        for p_id, path in enumerate(paths):
            tasks.append( executor.submit( extract_features_for_path, path) )
    out = np.concatenate([t.result() for t in tasks], axis = 0)

    # serial for debugging
    #out = []
    #for p_id, path in enumerate(paths):
    #    out.append( extract_features_for_path(path) )
    #out = np.concatenate(out, axis = 0)

    assert out.ndim == 2, str(out.shape)
    assert out.shape[0] == len(paths), str(out.shape)
    # TODO checkfor correct number of features

    return out
