
import vigra.graphs as graphs
import numpy as np


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

