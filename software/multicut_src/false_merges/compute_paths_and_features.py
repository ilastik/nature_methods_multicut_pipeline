import vigra
import vigra.graphs as graphs
import numpy as np
from concurrent import futures
#from scipy import interpolate

from ..ExperimentSettings import ExperimentSettings
#test
# calculate the distance transform for the given segmentation
def distance_transform(segmentation, anisotropy):
    edge_volume = np.concatenate(
            [vigra.analysis.regionImageToEdgeImage(segmentation[:,:,z])[:,:,None] for z in xrange(segmentation.shape[2])],
            axis = 2)
    dt = vigra.filters.distanceTransform(edge_volume, pixel_pitch=anisotropy, background=True)
    return dt


# TODO take nifty shortest paths and parallelize
def shortest_paths(
        indicator,
        pairs,
        n_threads = 1):
    """
    This function was copied from processing_lib.py
    :param indicator:
    :return:
    """

    gridgr         = graphs.gridGraph(indicator.shape)
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
def path_feature_aggregator(ds, paths):
    anisotropy_factor = ExperimentSettings().anisotropy_factor
    return np.concatenate([
        path_features_from_feature_images(ds, 0, paths, anisotropy_factor),
        path_features_from_feature_images(ds, 1, paths, anisotropy_factor),
        path_features_from_feature_images(ds, 2, paths, anisotropy_factor),
        features(paths, [1.,1.,anisotropy_factor])],
        axis = 1)



# don't cache for now
def path_features_from_feature_images(
        ds,
        inp_id,
        paths,
        anisotropy_factor):

    # FIXME for now we don't use fastfilters here
    feat_paths = ds.make_filters(inp_id, anisotropy_factor)
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
    stats = ExperimentSettings().feature_stats

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

    if len(paths) > 1:

        # We parallelize over the paths for now.
        # TODO parallelizing over filters might in fact be much faster, because
        # we avoid the single threaded i/o in the beginning!
        # it also lessens memory requirements if we have less threads than filters
        # parallel
        with futures.ThreadPoolExecutor(max_workers = ExperimentSettings().n_threads) as executor:
            tasks = []
            for p_id, path in enumerate(paths_in_roi):
                tasks.append( executor.submit( extract_features_for_path, path) )
            out = np.concatenate([t.result() for t in tasks], axis = 0)

    else:

        out = np.concatenate([extract_features_for_path(path) for path in paths_in_roi])


    # serial for debugging
    #out = []
    #for p_id, path in enumerate(paths_in_roi):
    #    out.append( extract_features_for_path(path) )
    #out = np.concatenate(out, axis = 0)

    assert out.ndim == 2, str(out.shape)
    assert out.shape[0] == len(paths), str(out.shape)
    # TODO checkfor correct number of features

    return out


def features(paths, anisotropy):



    def compute_path_length(path,aniso_temp=[1,1,1]):
        # TODO check that this actually agrees
        path_euclidean_diff =aniso_temp* np.diff(path, axis=0)
        path_euclidean_diff = np.sqrt(
            np.sum(np.square(path_euclidean_diff), axis=1))
        return np.sum(path_euclidean_diff, axis=0)


    def length(array, scale):
        # gibt einen array mit (nummer des pixels i,laenge des vektors von dem pixel i bis zum pixel i+scale) zurueck

        # TODO forgot to implement finite difference https://en.wikipedia.org/wiki/Finite_difference
        size = array.shape[0] - 3
        new = np.zeros((size, 2))

        for i in xrange(0, size):
            new[i, 0] = scale + i
            new[i, 1] = (np.linalg.norm(array[1 + i]) + np.linalg.norm(array[2 + i])) / 2

        return new


    def winkel(data):
        size = data.shape[0] - 1
        array = np.zeros(size)

        for i in xrange(0, size):
            x = data[i]
            y = data[i + 1]
            dot = np.dot(x, y)
            x_modulus = np.sqrt((x * x).sum())
            y_modulus = np.sqrt((y * y).sum())
            cos_angle = dot / x_modulus / y_modulus
            angle = np.arccos(cos_angle)  # Winkel in Bogenmas
            array[i] = angle

        array_dx2 = np.zeros(size - 2)

        for i in xrange(0, size - 2):
            array_dx2[i] = array[i] + array[2 + i] - 2 * array[1 + i]

        print "\n\n array_dx2: ", array_dx2,"\n\n"
        return array_dx2

    def curvature_berechnen(data):

        len1 = length(np.diff(data, axis=0), 0)
        array_winkel = winkel(np.diff(data, axis=0))
        len = np.abs(array_winkel / len1[:, 1] / len1[:, 1])  # dphi/ds
        len1[:, 1] = len
        return len1

    def maximum_ausgeben(path):


        maximum = np.amax(curvature_berechnen(path), axis=0)

        return maximum[1]




    aniso_temp = np.array(anisotropy)
    #
    # pathslist = []
    # for i in paths:
    #     pathslist.append(i)
    #
    # for number, data in enumerate(pathslist):
    #
    #     data = np.array([(elem1*aniso_temp[0], elem2*aniso_temp[1], elem3*aniso_temp[2]) for elem1, elem2, elem3 in data])
    #     data = data.transpose()
    #
    #    tck, u = interpolate.splprep(data, s=5000,k=3)
    #
    #    new = interpolate.splev(np.linspace(0, 1, len(data[0])), tck)
    #
    #     data = np.array(new).transpose()
    #     pathslist[number] = data
    #
    #
    #
    #
    # features_computed = np.concatenate([
    #     np.array([compute_path_length(path) for path in pathslist])[:, None],
    #     np.array([maximum_ausgeben(path) for path in pathslist])[:, None]],
    #     axis=1)
    #

    features_computed = np.array([compute_path_length(np.array(path), aniso_temp) for path in paths])[:,None]


    return features_computed

