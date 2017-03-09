
import vigra.graphs as graphs
import numpy as np
import vigra
from concurrent import futures
import h5py

def load_false_merges():
    return [], [], []


def shortest_paths(indicator, pairs, bounds=None, logger=None,
                   return_pathim=True, yield_in_bounds=False):
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

        if logger is not None:
            logger.logging('Calculating path from {} to {}', source, target)
        else:
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

        pathindices = np.swapaxes(path, 0, 1)
        if return_pathim:
            pathsim[pathindices[0], pathindices[1], pathindices[2]] = 1

    if return_pathim:
        if yield_in_bounds:
            return paths, pathsim, paths_in_bounds
        else:
            return paths, pathsim
    else:
        if yield_in_bounds:
            return paths, paths_in_bounds
        else:
            return paths


def path_features_from_feature_images(paths, feature_images, params):

    # For path length calculation
    # ---------------------------

    def compute_path_length(path, anisotropy):
        """
        Computes the length of a path

        :param path:
            np.array([[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [xm1, xm2, ..., xmn]])
            with n dimensions and m coordinates
        :param anisotropy: [a1, a2, ..., an]
        :return: path length (float)
        """

        pathlen = 0.
        for i in xrange(1, len(path)):

            add2pathlen = 0.
            for j in xrange(0, len(path[0, :])):
                add2pathlen += (anisotropy[j] * (path[i, j] - path[i - 1, j])) ** 2

            pathlen += add2pathlen ** (1. / 2)

        return pathlen

    # The path lengths only have to be computed once without using the vigra region features
    def compute_path_lengths(paths, anisotropy):

        path_lengths = []
        # for d, k, v, kl in paths.data_iterator():
        #     if type(v) is not type(paths):
        for path in paths:
            path_lengths.append(compute_path_length(np.array(path), anisotropy))

        return np.array(path_lengths)

    # And only do it when desired
    pathlength = False
    try:
        params.features.remove('Pathlength')
    except ValueError:
        # Means that 'Pathlength' was not in the list
        pass
    else:
        # 'Pathlength' was in the list and is now successfully removed
        pathlength = True

    with open(params.feat_list_file, 'r') as f:
        import pickle
        feat_list = pickle.load(f)[params.experiment_key]

    # Iterate over the paths
    # ----------------------

    newfeats = dict()
    feats_array = np.array([])
    anisotropy = params.anisotropy

    for p_id, path in enumerate(paths):

        new_feats_array = np.array([])

        print 'Working on path {} of {}'.format(p_id + 1, len(paths))

        # Create some working image with a path in it
        path_image = np.zeros(feature_images['segmentation'].shape, dtype=np.uint32)
        path_sa = np.swapaxes(path, 0, 1)
        path_image[path_sa[0], path_sa[1], path_sa[2]] = 1

        # TODO: Pre-compute the region features and then sort them into the array
        # The parallelized way
        with futures.ThreadPoolExecutor(params.max_threads) as do_stuff:

            print 'Starting thread pool with {} threads'.format(params.max_threads)

            tasks = {}
            done_list = []

            def extract_region_features_wrapper(source_feature, id, path_image):
                featim = feature_images[source_feature].get_feature(id)
                return vigra.analysis.extractRegionFeatures(
                    featim.astype(np.float32),
                    path_image, ignoreLabel=0,
                    features=params.features
                )

            for feat in feat_list:

                if feat[0] == 'Pathlength': continue

                if str.join('/', feat[:-1]) not in done_list:

                    done_list.append(str.join('/', feat[:-1]))
                    print 'Computing {}'.format(str.join('/', feat[:-1]))

                    tasks[str.join('/', feat[:-1])] = (
                        do_stuff.submit(
                            extract_region_features_wrapper,
                            feat[0], str.join('/', feat[1:-1]), path_image
                        )
                    )

            # for k, v in tasks:
            #     region_features[k] = v.result()

            region_features = {k: v.result() for k, v in tasks.iteritems()}

        # TODO: For each feature image extract the region features
        # TODO: Use featlist.pkl to determine the order and directly create a feature array for the path
        # Loop over featlist (featlist.pkl) and get the respective feature image by
        for feat in feat_list:

            # print '    Working on feat = {}'.format(feat)

            if feat[0] == 'Pathlength':

                if not pathlength:
                    raise ValueError('Pathlength was found in feature order list but not in feature list')

                # # Calculate path length
                # if not 'Pathlength' in newfeats.keys():
                #     newfeats['Pathlength'] = np.array([compute_path_length(path, anisotropy)])
                # else:
                #     newfeats['Pathlength'] = np.concatenate((
                #         newfeats['Pathlength'], [compute_path_length(path, anisotropy)]))

                # Append the current feature to the list
                new_feats_array = np.concatenate((new_feats_array, [compute_path_length(path, anisotropy)]))

            else:

                # source_feature = feat[0]
                #
                # # The non-parallel way
                # if str.join('/', feat[:-1]) not in region_features.keys():
                #
                #     # Compute the region features of the corresponding path and image here
                #
                #     featim = feature_images[source_feature].get_feature(str.join('/', feat[1:-1]))
                #
                #     # Extract the region features of the working image
                #     region_features[str.join('/', feat[:-1])] = vigra.analysis.extractRegionFeatures(
                #         featim.astype(np.float32),
                #         path_image, ignoreLabel=0,
                #         features=params.features
                #     )

                # Append the current feature to the list
                if region_features[str.join('/', feat[:-1])][str(feat[-1])][1:].ndim > 1:
                    new_feats_array = np.concatenate((new_feats_array, region_features[str.join('/', feat[:-1])][str(feat[-1])][1:].squeeze()))
                else:
                    new_feats_array = np.concatenate((new_feats_array, region_features[str.join('/', feat[:-1])][str(feat[-1])][1:]))

                # # Take the region feature as defined in 'feat' and append it to the feature array
                # if not str.join('/', feat[:-1]) in newfeats.keys():
                #     newfeats[str.join('/', feat[:-1])] = dict()
                # if not feat[-1] in newfeats[str.join('/', feat[:-1])].keys():
                #     newfeats[str.join('/', feat[:-1])][str(feat[-1])] = region_features[str.join('/', feat[:-1])][str(feat[-1])][1:]
                # else:
                #     newfeats[str.join('/', feat[:-1])][str(feat[-1])] = np.concatenate((
                #         newfeats[str.join('/', feat[:-1])][str(feat[-1])], region_features[str.join('/', feat[:-1])][str(feat[-1])][1:]
                #     ))

        if not feats_array.any():
            feats_array = np.array([new_feats_array])
        else:
            feats_array = np.concatenate((feats_array, [new_feats_array]), axis=0)


    return feats_array
