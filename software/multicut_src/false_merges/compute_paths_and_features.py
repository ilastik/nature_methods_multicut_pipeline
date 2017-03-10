import vigra
import vigra.graphs as graphs
import numpy as np

from .. import DataSet
from .. import cacher_hdf5

def shortest_paths(indicator,
        pairs,
        bounds=None,
        yield_in_bounds=False):
    """
    This function was copied from processing_lib.py
    :param indicator:
    :param pairs:
    :param bounds:
    :param logger:
    :param return_pathim:
    :param yield_in_bounds:
    :return:
    """

    # Crate the grid graph and shortest path objects
    gridgr = graphs.gridGraph(indicator.shape)
    indicator = indicator.astype(np.float32)
    gridgr_edgeind = graphs.edgeFeaturesFromImage(gridgr, indicator)
    instance = graphs.ShortestPathPathDijkstra(gridgr)

    # Initialize paths image
    if return_pathim:
        pathsim = np.zeros(indicator.shape)
    # Initialize list of path coordinates
    paths = []
    if yield_in_bounds:
        paths_in_bounds = []

    for pair in pairs:

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
                paths.append(path + [bounds[0].start, bounds[1].start, bounds[2].start])
                if yield_in_bounds:
                    paths_in_bounds.append(path)
            else:
                paths.append(path)

    if yield_in_bounds:
        return paths, paths_in_bounds
    else:
        return paths


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
        pathlen = 0.
        #for i in xrange(1, len(path)):
        #    add2pathlen = 0.
        #    for j in xrange(0, len(path[0, :])):
        #        add2pathlen += (anisotropy[j] * (path[i, j] - path[i - 1, j])) ** 2

        #    pathlen += add2pathlen ** (1. / 2)
        # TODO check that this actually agrees
        paths_euclidean_diffs = np.prod( aniso_temp, np.diff(path) ) )
        paths_euclidean_diffs = np.sqrt(
                np.square( np.sum(paths_euclidean_diffs,axis=) ) )
        return np.sum(paths_euclidean_diffs, axis=0)
    aniso_temp = np.array(anisotropy)[None,:] # TODO is this the correct brodcasting?
    return np.array([compute_path_length(np.array(path), aniso_temp) for path in paths])


# don't cache for now
# TODO enable bounded paths
def path_features_from_feature_images(
        ds,
        inp_id,
        paths,
        params,
        anisotropy_factor):

    feat_paths = ds.make_filters(inp_id, anisotropy_factor)
    # TODO sort the feat_path correctly
    # load the feature images -> FIXME this might be to memory hungry
    # TODO for bounded path, only load the subblock with h5py
    feature_volumes = [ vigra.readHDF5(pp, 'data') for pp in feat_paths ]
    stats = params.features

    def extract_features_for_path(path):

        path_image = np.zeros(shape, dtype='uint32')
        path_sa = np.swapaxes(path, 0, 1)
        path_image[path_sa[0], path_sa[1], path_sa[2]] = 1

        path_features = []
        for feature_volume in feature_volumes:
            extractor = vigra.analysis.extractRegionFeatures(
                    feature_volume,
                    path_image,
                    ignoreLabel = 0,
                    features = stats)
            path_features.append( extractor[stat] for stat in stats) # TODO make sure that dimensions match!
        return np.array(path_features)

    with futures.ThreadPoolExecutor(params.max_threads) as executor:
        tasks = []
        for p_id, path in enumerate(paths):
            tasks.append( executor.submit( extract_region_features_wrapper(
                extract_features_for_path, path) ) )
    return np.concatenate([t.result() for t in tasks], axis = 0)
