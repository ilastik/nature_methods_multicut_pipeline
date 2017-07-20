import numpy as np
import vigra
import os
import cPickle as pickle
import shutil
import itertools
import h5py
import numpy as np
import vigra
import cPickle as pickle
from copy import deepcopy
from skimage.morphology import skeletonize_3d
from Queue import LifoQueue
from time import time
import h5py
import nifty_with_cplex as nifty
from skimage.measure import label
from time import time

# relative imports from top level dir
from ..MCSolverImpl   import probs_to_energies
from ..Postprocessing import remove_small_segments
from ..lifted_mc import compute_and_save_long_range_nh, optimize_lifted, compute_and_save_lifted_nh
from ..EdgeRF import RandomForest
from ..ExperimentSettings import ExperimentSettings
from ..tools import find_matching_row_indices

# imports from this dir
from .compute_paths_and_features import shortest_paths, distance_transform, path_feature_aggregator
from .compute_border_contacts import compute_path_end_pairs, compute_path_end_pairs_and_labels, compute_border_contacts_old, compute_border_contacts


def close_cavities(init_volume):
    """close cavities in segments so skeletonization don't bugs"""

    print "looking for open cavities inside the object..."
    volume=deepcopy(init_volume)
    volume[volume==0]=2
    lab=label(volume)

    if len(np.unique(lab))==2:
        print "No cavities to close!"
        return init_volume

    count,what=0,0

    for uniq in np.unique(lab):
        if len(np.where(lab == uniq)[0])> count:
            count=len(np.where(lab == uniq)[0])
            what=uniq

    volume[lab==what]=0
    volume[lab != what] = 1

    print "cavities closed"

    return volume


def check_box(volume, point, is_queued_map, is_node_map, stage=1):
    """checks the Box around the point for points which are 1,
    but were not already put in the queue and returns them in a list"""
    list_not_queued = []
    list_are_near = []
    list_is_node = []

    for x in xrange(-1, 2):

        # Edgecase for x
        if point[0] + x < 0 or point[0] + x > volume.shape[0] - 1:
            continue

        for y in xrange(-1, 2):

            # Edgecase for y
            if point[1] + y < 0 or point[1] + y > volume.shape[1] - 1:
                continue

            for z in xrange(-1, 2):

                # Edgecase for z
                if point[2] + z < 0 or point[2] + z > volume.shape[2] - 1:
                    continue

                # Dont look at the middle point
                if x == 0 and y == 0 and z == 0:
                    continue

                if volume[point[0] + x, point[1] + y, point[2] + z] > 0:

                    list_are_near.extend([[point[0] + x, point[1] + y, point[2] + z]])

                    if is_queued_map[point[0] + x, point[1] + y, point[2] + z] == 0:
                        list_not_queued.extend([[point[0] + x, point[1] + y, point[2] + z]])


                    if is_node_map[point[0] + x, point[1] + y, point[2] + z] != 0:
                        list_is_node.extend([[point[0] + x, point[1] + y, point[2] + z]])

    return list_not_queued, list_is_node, list_are_near


def init(volume):
    """searches for the first node to start with"""
    if len(np.where(volume)[0]) == 0:
        return np.array([-1, -1, -1])
    point = np.array((np.where(volume)[:][0][0], np.where(volume)[:][1][0], np.where(volume)[:][2][0]))

    is_queued_map = np.zeros(volume.shape, dtype=int)
    is_queued_map[point[0], point[1], point[2]] = 1

    not_queued, _, _ = check_box(volume, point, is_queued_map, np.zeros(volume.shape, dtype=int))

    if len(not_queued) == 2:
        while True:
            point = np.array(not_queued[0])
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            not_queued, _, _ = check_box(volume, point, is_queued_map, np.zeros(volume.shape, dtype=int))

            if len(not_queued) != 1:
                break

    return point


def stage_one(img, dt):
    """stage one, finds all nodes and edges, except for loops"""

    # initializing
    volume = deepcopy(img)
    is_queued_map = np.zeros(volume.shape, dtype=int)
    is_node_map = np.zeros(volume.shape, dtype=int)
    is_term_map = np.zeros(volume.shape, dtype=int)
    is_branch_map = np.zeros(volume.shape, dtype=int)
    is_standart_map = np.zeros(volume.shape, dtype=int)
    nodes = {}
    edges = []
    last_node = 1
    current_node = 1
    queue = LifoQueue()
    point = init(volume)
    loop_list = []
    branch_point_list = []
    node_list = []
    length = 0
    if (point == np.array([-1, -1, -1])).all():
        return is_node_map, is_term_map, is_branch_map, nodes, edges

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued, is_node_list, are_near = check_box(volume, point, is_queued_map, is_node_map)
    nodes[current_node] = point

    while len(not_queued) == 0:
        volume[point[0], point[1], point[2]] = 0
        is_queued_map[point[0], point[1], point[2]] = 0
        nodes = {}
        point = init(volume)
        if (point == np.array([-1, -1, -1])).all():
            return is_node_map, is_term_map, is_branch_map, nodes, edges
        is_queued_map[point[0], point[1], point[2]] = 1
        not_queued, is_node_list, are_near = check_box(volume, point, is_queued_map, is_node_map)
        nodes[current_node] = point

    for i in not_queued:
        queue.put(np.array([i, current_node, length,
                            [[point[0], point[1], point[2]]],
                            [dt[point[0], point[1], point[2]]]]))
        is_queued_map[i[0], i[1], i[2]] = 1

    if len(not_queued) == 1:
        is_term_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node

    else:
        is_branch_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node

    while queue.qsize():

        # pull item from queue
        point, current_node, length, edge_list, dt_list = queue.get()

        not_queued, is_node_list, are_near = check_box(volume, point, is_queued_map, is_node_map)

        # if current_node==531:
        #     print "hi"
        #     print "hi"

        # standart point
        if len(not_queued) == 1:
            dt_list.extend([dt[point[0], point[1], point[2]]])
            edge_list.extend([[point[0], point[1], point[2]]])
            length = length + np.linalg.norm(
                [point[0] - not_queued[0][0], point[1] - not_queued[0][1], (point[2] - not_queued[0][2]) * 10])
            queue.put(np.array([not_queued[0], current_node, length, edge_list, dt_list]))
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            branch_point_list.extend([[point[0], point[1], point[2]]])
            is_standart_map[point[0], point[1], point[2]] = 1

        elif len(not_queued) == 0 and (len(are_near) > 1 or len(is_node_list) > 0):
            loop_list.extend([current_node])


        # terminating point
        elif len(not_queued) == 0 and len(are_near) == 1 and len(is_node_list) == 0:
            last_node = last_node + 1
            nodes[last_node] = point
            dt_list.extend([dt[point[0], point[1], point[2]]])
            edge_list.extend([[point[0], point[1], point[2]]])
            node_list.extend([[point[0], point[1], point[2]]])
            edges.extend([[np.array([current_node, last_node]), length, edge_list, dt_list]])
            is_term_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node




        # branch point
        elif len(not_queued) > 1:
            dt_list.extend([dt[point[0], point[1], point[2]]])
            edge_list.extend([[point[0], point[1], point[2]]])
            last_node = last_node + 1
            nodes[last_node] = point
            # build edge
            edges.extend([[np.array([current_node, last_node]), length, edge_list, dt_list]])  # build edge
            node_list.extend([[point[0], point[1], point[2]]])
            # putting node branches in the queue
            for x in not_queued:
                length = np.linalg.norm([point[0] - x[0], point[1] - x[1], (point[2] - x[2]) * 10])
                queue.put(np.array([x, last_node, length,
                                    [[point[0], point[1], point[2]]],
                                    [dt[point[0], point[1], point[2]]]]))
                is_queued_map[x[0], x[1], x[2]] = 1

            is_branch_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node

    # if (len(np.where(volume)[0]) - len(np.where(is_branch_map)[0]) - len(np.where(is_term_map)[0]) - len(np.where(is_standart_map)[0]))!=0:
    #     pass
    #     print "assert"
    # else:
    #     print "no assert"

    # assert((len(np.where(volume)[0]) - len(np.where(is_branch_map)[0]) - len(np.where(is_term_map)[0]) - len(np.where(is_standart_map)[0]))==0), "too few points were looked at/some were looked at twice !"



    return is_node_map, is_term_map, is_branch_map, nodes, edges, loop_list


def stage_two(is_node_map, is_term_map, edges, dt):
    """finds edges for loops"""

    list_term = np.array(np.where(is_term_map)).transpose()

    for point in list_term:
        _, _, list_near_nodes = check_box(is_node_map, point, np.zeros(is_node_map.shape, dtype=int),
                                          np.zeros(is_node_map.shape, dtype=int), 2)

        assert (len(list_near_nodes) == 0)

        #     if len(list_near_nodes) != 0:
        #
        #         assert()
        #         node_number=is_term_map[point[0], point[1], point[2]]
        #         is_term_map[point[0], point[1], point[2]]=0
        #         print "hi"
        #
        #     for i in list_near_nodes:
        #         edge_list = []
        #         edge_list.extend([[point[0], point[1], point[2]]])
        #         edge_list.extend([[i[0], i[1], i[2]]])
        #         dt_list = []
        #         dt_list.extend([dt[point[0], point[1], point[2]]])
        #         dt_list.extend([dt[i[0], i[1], i[2]]])
        #         edges.extend([[np.array([is_node_map[point[0],point[1],point[2]],
        #                                  is_node_map[i[0],i[1],i[2]]]),
        #                        np.linalg.norm([point[0] - i[0], point[1] - i[1],
        #                                        (point[2] - i[2]) * 10]),
        #                        edge_list,
        #                        dt_list]]) #build edge
        #
        #
        # return edges,is_term_map


def form_term_list(is_term_map):
    """returns list of terminal points taken from an image"""

    term_where = np.array(np.where(is_term_map)).transpose()
    term_list = []
    for point in term_where:
        term_list.extend([is_term_map[point[0], point[1], point[2]]])
    term_list = np.array([term for term in term_list])

    return term_list


def skeleton_to_graph(img, dt):
    """main function, wraps up stage one and two"""

    time_before_stage_one_1 = time()
    is_node_map, is_term_map, is_branch_map, nodes, edges, loop_list = stage_one(img, dt)
    if len(nodes) == 0:
        return nodes, np.array(edges), [], is_node_map

    edges, is_term_map = stage_two(is_node_map, is_term_map, edges, dt)

    edges = [[a, b, c, max(d)] for a, b, c, d in edges]

    term_list = form_term_list(is_term_map)
    term_list -= 1
    # loop_list -= 1
    return nodes, np.array(edges), term_list, is_node_map, loop_list


def get_unique_rows(array, return_index=False):
    """ make the rows of array unique
        see http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """

    array_view = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    _, idx = np.unique(array_view, return_index=True)
    unique_rows = array[idx]
    if return_index:
        return unique_rows, idx
    else:
        return unique_rows


def unique_rows(a):
    """same same but different"""

    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def graph_and_edge_weights(nodes, edges_and_lens):
    """creates graph from edges and nodes, length of edges included"""

    edges = []
    edge_lens = []
    edges.extend(edges_and_lens[:, 0])
    edges = np.array(edges, dtype="uint32")
    edge_lens.extend(edges_and_lens[:, 1])
    edge_lens = np.array([edge for edge in edge_lens])
    edges = np.sort(edges, axis=1)
    # remove duplicates from edges and edge-lens
    edges, unique_idx = get_unique_rows(edges, return_index=True)
    edge_lens = edge_lens[unique_idx]
    edges_and_lens = edges_and_lens[unique_idx]
    edges_and_lens[:, 0] -= 1

    assert len(edges) == len(edge_lens)
    assert edges.shape[1] == 2
    node_list = np.array(nodes.keys())
    edges = np.array(edges, dtype='uint32')
    edges = np.sort(edges, axis=1)
    edges -= 1
    n_nodes = edges.max() + 1
    assert len(node_list) == n_nodes
    g = nifty.graph.UndirectedGraph(n_nodes)
    g.insertEdges(edges)
    assert g.numberOfEdges == len(edge_lens), '%i, %i' % (g.numberOfEdges, len(edge_lens))
    return g, edge_lens, edges_and_lens


#
def check_connected_components(g):
    """check that we have the correct number of connected components"""

    cc = nifty.graph.components(g)
    cc.build()

    components = cc.componentLabels()

    # print components
    n_ccs = len(np.unique(components))
    assert n_ccs == 1, str(n_ccs)


def edge_paths_and_counts_for_nodes(g, weights, node_list, n_threads=8):
    """
    Returns the path of edges for all pairs of nodes in node list as
    well as the number of times each edge is included in a shortest path.
    @params:
    g         : nifty.graph.UndirectedGraph, the underlying graph
    weights   : list[float], the edge weights for shortest paths
    node_list : list[int],   the list of nodes that will be considered for the shortest paths
    n_threads : int, number of threads used
    @returns:
    edge_paths: dict[tuple(int,int),list[int]] : Dictionary with list of edge-ids corresponding to the
                shortest path for each pair in node_list
    edge_counts: np.array[int] : Array with number of times each edge was visited in a shortest path
    """

    edge_paths = {}
    edge_counts = np.zeros(g.numberOfEdges, dtype='uint32')

    # single threaded implementation
    if n_threads < 2:

        # build the nifty shortest path object
        path_finder = nifty.graph.ShortestPathDijkstra(g)

        # iterate over the source nodes
        # we don't need to go to the last node, because it won't have any more targets
        for ii, u in enumerate(node_list[:-1]):

            # target all nodes in node list tat we have not visited as source
            # already (for these the path is already present)
            target_nodes = node_list[ii + 1:]

            # find the shortest path from source node u to the target nodes
            shortest_paths = path_finder.runSingleSourceMultiTarget(
                weights.tolist(),
                u,
                target_nodes,
                returnNodes=False
            )
            assert len(shortest_paths) == len(target_nodes)

            # extract the shortest path for each node pair and
            # increase the edge counts
            for jj, sp in enumerate(shortest_paths):
                v = target_nodes[jj]
                edge_paths[(u, v)] = sp
                edge_counts[sp] += 1

    # multi-threaded implementation
    # this might be quite memory hungry!
    else:

        # construct the target nodes for all source nodes and run shortest paths
        # in parallel, don't need last node !
        all_target_nodes = [node_list[ii + 1:] for ii in xrange(len(node_list[:-1]))]
        all_shortest_paths = nifty.graph.shortestPathMultiTargetParallel(
            g,
            weights.tolist(),
            node_list[:-1],
            all_target_nodes,
            returnNodes=False,
            numberOfThreads=n_threads
        )
        assert len(all_shortest_paths) == len(node_list) - 1, "%i, %i" % (len(all_shortest_paths), len(node_list) - 1)

        # TODO this is still quite some serial computation overhead.
        # for good paralleliztion, this should also be parallelized

        # extract the shortest paths for all node pairs and edge counts
        for ii, shortest_paths in enumerate(all_shortest_paths):

            u = node_list[ii]
            target_nodes = all_target_nodes[ii]
            for jj, sp in enumerate(shortest_paths):
                v = target_nodes[jj]
                edge_paths[(u, v)] = sp
                edge_counts[sp] += 1

    return edge_paths, edge_counts


def check_edge_paths(edge_paths, node_list):
    """checks edge paths (constantin)"""

    from itertools import combinations
    pairs = combinations(node_list, 2)
    pair_list = [pair for pair in pairs]

    # make sure that we have all combination in the edge_paths
    for pair in pair_list:
        assert pair in edge_paths

    # make sure that we don't have any spurious pairs in edge_paths
    for pair in edge_paths:
        assert pair in pair_list

    print "passed"


def compute_graph_and_paths(img, dt, modus="run"):
    """ overall wrapper for all functions, input: label image; output: paths
        sampled from skeleton
    """

    nodes, edges, term_list, is_node_map, loop_list = skeleton_to_graph(img, dt)
    if len(term_list) == 0:
        return []
    g, edge_lens, edges = graph_and_edge_weights(nodes, edges)

    check_connected_components(g)

    loop_uniq, loop_nr = np.unique(loop_list, return_counts=True)

    for where in np.where(loop_nr > 1)[0]:

        adjacency = np.array([[adj_node, adj_edge] for adj_node, adj_edge
                              in g.nodeAdjacency(loop_uniq[where] - 1)])

        if (len(adjacency)) == 1:
            term_list = np.append(term_list, loop_uniq[where] - 1)

    # if modus=="testing":
    #     return term_list,edges,g,nodes


    edge_paths, edge_counts = edge_paths_and_counts_for_nodes(g,
                                                              edge_lens,
                                                              term_list[:30], 8)
    check_edge_paths(edge_paths, term_list[:30])
    edge_paths_julian = {}

    for pair in edge_paths.keys():
        edge_paths_julian[pair] = []

        for idx in edge_paths[pair]:
            edge_paths_julian[pair].extend(edges[idx][2])

    final_edge_paths = {}

    for pair in edge_paths_julian.keys():
        final_edge_paths[pair] = unique_rows(edge_paths_julian[pair])

    workflow_paths = []

    # for workflow functions
    for pair in final_edge_paths.keys():
        workflow_paths.extend([final_edge_paths[pair]])

    if modus == "testing":
        return term_list, edges, g, nodes

    return workflow_paths


def cut_off(all_paths_unfinished,paths_to_objs_unfinished,
            cut_off_array,ratio_true=0.13,ratio_false=0.4):
    """ cuts array so that all paths with a ratio between ratio_true% and
        ratio_false% are not in the paths as they are not clearly to identify
        as false paths or true paths
    """

    print "start cutting off array..."
    test_label = []
    con_label = {}
    test_length = []
    con_len = {}

    for label in cut_off_array.keys():
        #FIXME  in cut_off conc=np.concatenate(con_label[label]).tolist() ValueError: need at least one array to concatenate
        if len(cut_off_array[label])==0:
            continue
        con_label[label]=[]
        con_len[label]=[]
        for path in cut_off_array[label]:
            test_label.append(path[0])
            con_label[label].append(path[0])
            test_length.append(path[1])
            con_len[label].append(path[1])

    help_array=[]
    for label in con_label.keys():

        conc=np.concatenate(con_label[label]).tolist()
        counter=[0,0]
        for number in np.unique(conc):
            many = conc.count(number)

            if counter[1] < many:
                counter[0] = number
                counter[1] = many
        #TODO con label len =0
        for i in xrange(0,len(con_label[label])):
            help_array.extend([counter[0]])

    end = []

    for idx, path in enumerate(test_label):


        overall_length = 0
        for i in test_length[idx]:
            overall_length = overall_length + i

        less_length = 0
        for u in np.where(np.array(path) != help_array[idx]):
            for index in u:
                less_length = less_length + test_length[idx][index]

        end.extend([less_length/ overall_length])



    path_classes=[]
    all_paths=[]
    paths_to_objs=[]
    for idx,ratio in enumerate(end):
        if ratio<ratio_true:
            path_classes.extend([True])
            all_paths.extend([all_paths_unfinished[idx]])
            paths_to_objs.extend([paths_to_objs_unfinished[idx]])

        elif ratio>ratio_false:
            path_classes.extend([False])
            all_paths.extend([all_paths_unfinished[idx]])
            paths_to_objs.extend([paths_to_objs_unfinished[idx]])


    print "finished cutting of"

    return np.array(all_paths), np.array(paths_to_objs, dtype="float64"),\
           np.array(path_classes)


def extract_paths_from_segmentation_alex(
        ds,
        seg_path,
        key,
        paths_cache_folder=None):
    """
        extract paths from segmentation, for pipeline
    """


    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s.h5' % ds.ds_name)
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array( [ path.reshape( (len(path)/3, 3) ) for path in all_paths ] )
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')

    # otherwise compute the paths

    else:

        seg = vigra.readHDF5(seg_path, key)
        gt = deepcopy(seg)
        img = deepcopy(seg)
        all_paths = []
        paths_to_objs = []


        cut_off_array = {}
        len_uniq=len(np.unique(seg))-1
        for idx,label in enumerate(np.unique(seg)):
            print "Number ", idx, " without labels of ",len_uniq-1
            if label == 0:
                continue


            # masking volume
            img[seg != label] = 0
            img[seg == label] = 1

            # no skeletons too close to the borders No.2
            #img[dt == 0] = 0

            # skeletonize
            skel_img = skeletonize_3d(img)

            paths=compute_graph_and_paths(skel_img)

            percentage = []
            len_path=len(paths)
            for idx,path in enumerate(paths):
                print idx ,". path of ",len_path-1
                #TODO better workaround
                # workaround till tomorrow
                workaround_array=[]
                length_array=[]
                last_point=path[0]
                for i in path:
                    workaround_array.extend([gt[i[0],i[1],i[2]]])
                    length_array.extend([np.linalg.norm([last_point[0] - i[0],
                                                      last_point[1] - i[1], (last_point[2] - i[2]) * 10])])
                    last_point=i


                all_paths.extend([path])
                paths_to_objs.extend([label])

                half_length_array=[]

                for idx,obj in enumerate(length_array[:-1]):
                    half_length_array.extend([length_array[idx]/2+length_array[idx+1]/2])
                half_length_array.extend([length_array[-1] / 2])


                percentage.extend([[workaround_array, half_length_array]])

            cut_off_array[label] = percentage

        all_paths, paths_to_objs,_ =cut_off(all_paths,paths_to_objs,cut_off_array)

        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')

    return all_paths, paths_to_objs



def extract_paths_and_labels_from_segmentation_alex(
        ds,
        seg,
        seg_id,
        gt,
        correspondence_list,
        paths_cache_folder=None):

    """
        extract paths from segmentation, for learning
    """

    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s_seg_%i.h5' % (ds.ds_name, seg_id))
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array( [ path.reshape( (len(path)/3, 3) ) for path in all_paths ] )
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')
        path_classes = vigra.readHDF5(paths_save_file, 'path_classes')
        correspondence_list = vigra.readHDF5(paths_save_file, 'correspondence_list').tolist()

    # otherwise compute paths
    else:

        img = deepcopy(seg)
        all_paths = []
        paths_to_objs = []


        cut_off_array = {}
        len_uniq=len(np.unique(seg))-1
        for label in np.unique(seg):
            print "Label ", label, " of ",len_uniq
            if label == 0:
                continue

            # masking volume
            img[seg != label] = 0
            img[seg == label] = 1


            # skeletonize
            skel_img = skeletonize_3d(img)

            paths=compute_graph_and_paths(skel_img)

            percentage = []

            for path in paths:

                #TODO better workaround
                # workaround till tomorrow
                workaround_array=[]
                length_array=[]
                last_point=path[0]
                for i in path:
                    workaround_array.extend([gt[i[0],i[1],i[2]]])
                    length_array.extend([np.linalg.norm([last_point[0] - i[0],
                                                      last_point[1] - i[1], (last_point[2] - i[2]) * 10])])
                    last_point=i


                all_paths.extend([path])
                paths_to_objs.extend([label])

                half_length_array=[]

                for idx,obj in enumerate(length_array[:-1]):
                    half_length_array.extend([length_array[idx]/2+length_array[idx+1]/2])
                half_length_array.extend([length_array[-1] / 2])


                percentage.extend([[workaround_array, half_length_array]])

            cut_off_array[label] = percentage

        all_paths,paths_to_objs,path_classes = cut_off(all_paths, paths_to_objs, cut_off_array)

        # if caching is enabled, write the results to cache
        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                print 'Saving paths in {}'.format(paths_save_file)
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')
            vigra.writeHDF5(path_classes, paths_save_file, 'path_classes')
            vigra.writeHDF5(correspondence_list, paths_save_file, 'correspondence_list')

    return all_paths, paths_to_objs, path_classes, correspondence_list


def extract_paths_from_segmentation_dump(
        ds,
        seg_path,
        key,
        paths_cache_folder = None):

    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s.h5' % ds.ds_name)
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array( [ path.reshape( (len(path)/3, 3) ) for path in all_paths ] )
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')

    # otherwise compute the paths
    else:
        # TODO we don't remove small objects for now, because this would relabel the segmentation, which we don't want in this case
        seg = vigra.readHDF5(seg_path, key)
        dt = ds.inp(ds.n_inp-1) # we assume that the last input is the distance transform

        # Compute path end pairs
        # TODO debug the new border contact computation, which is much faster
        #border_contacts = compute_border_contacts(seg, False)
        border_contacts = compute_border_contacts_old(seg, dt)

        path_pairs, paths_to_objs = compute_path_end_pairs(border_contacts)
        # Sort the paths_to_objs by size (not doing that leads to a possible bug in the next loop)
        order = np.argsort(paths_to_objs)
        paths_to_objs = np.array(paths_to_objs)[order]
        path_pairs = np.array(path_pairs)[order]

        # Invert the distance transform and take penalty power
        dt = np.amax(dt) - dt
        dt = np.power(dt, ExperimentSettings().paths_penalty_power)

        all_paths = []
        for obj in np.unique(paths_to_objs):

            # Mask distance transform to current object
            masked_dt = dt.copy()
            masked_dt[seg != obj] = np.inf

            # Take only the relevant path pairs
            pairs_in = path_pairs[paths_to_objs == obj]

            paths = shortest_paths(masked_dt,
                    pairs_in,
                    n_threads = ExperimentSettings().n_threads)
            # paths is now a list of numpy arrays
            all_paths.extend(paths)

        # Remove all paths that are None, i.e. were initially not computed or were subsequently removed
        keep_mask = np.array([isinstance(x, np.ndarray) for x in all_paths], dtype = np.bool)
        all_paths = np.array(all_paths)[keep_mask]
        paths_to_objs = paths_to_objs[keep_mask]

        # if we cache paths save the results
        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')

    return all_paths, paths_to_objs


def extract_paths_and_labels_from_segmentation_dump(
        ds,
        seg,
        seg_id,
        gt,
        correspondence_list,
        paths_cache_folder = None):
    """
    params:
    """

    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s_seg_%i.h5' % (ds.ds_name, seg_id))
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array( [ path.reshape( (len(path)/3, 3) ) for path in all_paths ] )
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')
        path_classes = vigra.readHDF5(paths_save_file, 'path_classes')
        correspondence_list = vigra.readHDF5(paths_save_file, 'correspondence_list').tolist()

    # otherwise compute paths
    else:
        assert seg.shape == gt.shape
        dt = ds.inp(ds.n_inp-1) # we assume that the last input is the distance transform

        # Compute path end pairs
        # TODO debug the new border contact computation, which is much faster
        #border_contacts = compute_border_contacts(seg, False)
        border_contacts = compute_border_contacts_old(seg, dt)

        # This is supposed to only return those pairs that will be used for path computation
        # TODO: Throw out some under certain conditions (see also within function)
        path_pairs, paths_to_objs, path_classes, path_gt_labels, correspondence_list = compute_path_end_pairs_and_labels(
            border_contacts, gt, correspondence_list
        )

        # Invert the distance transform and take penalty power
        dt = np.amax(dt) - dt
        dt = np.power(dt, ExperimentSettings().paths_penalty_power)

        all_paths = []
        for obj in np.unique(paths_to_objs):

            # Mask distance transform to current object
            # TODO use a mask in dijkstra instead
            masked_dt = dt.copy()
            masked_dt[seg != obj] = np.inf

            # Take only the relevant path pairs
            pairs_in = path_pairs[paths_to_objs == obj]

            paths = shortest_paths(masked_dt,
                    pairs_in,
                    n_threads = ExperimentSettings().n_threads)
            # paths is now a list of numpy arrays
            all_paths.extend(paths)


        # TODO: Here we have to ensure that every path is actually computed
        # TODO:  --> Throw not computed paths out of the lists

        # TODO: Remove paths under certain criteria
        # TODO: Do this only if GT is supplied
        # a) Class 'non-merged': Paths cross labels in GT multiple times
        # b) Class 'merged': Paths have to contain a certain amount of pixels in both GT classes
        # TODO implement stuff here

        # Remove all paths that are None, i.e. were initially not computed or were subsequently removed
        keep_mask = np.array( [isinstance(x,np.ndarray) for x in all_paths], dtype = np.bool )
        all_paths = np.array(all_paths)[keep_mask]
        paths_to_objs = paths_to_objs[keep_mask]
        path_classes  = path_classes[keep_mask]

        # if caching is enabled, write the results to cache
        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                print 'Saving paths in {}'.format(paths_save_file)
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')
            vigra.writeHDF5(path_classes, paths_save_file, 'path_classes')
            vigra.writeHDF5(correspondence_list, paths_save_file, 'correspondence_list')

    return all_paths, paths_to_objs, path_classes, correspondence_list


# cache the random forest here
def train_random_forest_for_merges(
        trainsets, # list of datasets with training data
        mc_segs_train, # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        rf_cache_folder=None,
        paths_cache_folder=None
):

    if rf_cache_folder is not None:
        if not os.path.exists(rf_cache_folder):
            os.mkdir(rf_cache_folder)

    rf_save_path = '' if rf_cache_folder is None else os.path.join(
        rf_cache_folder,
        'rf_merges_%s' % '_'.join([ds.ds_name for ds in trainsets])
    ) # TODO more meaningful save name

    # check if rf is already cached
    if RandomForest.is_cached(rf_save_path):
        print "Loading rf from:", rf_save_path
        rf = RandomForest.load_from_file(rf_save_path, 'rf', ExperimentSettings().n_threads)

    # otherwise do the actual calculations
    else:
        features_train = []
        labels_train = []

        # loop over the training datasets
        for ds_id, paths_to_betas in enumerate(mc_segs_train):

            current_ds = trainsets[ds_id]
            keys_to_betas = mc_segs_train_keys[ds_id]
            assert len(keys_to_betas) == len(paths_to_betas), "%i, %i" % (len(keys_to_betas), len(paths_to_betas))

            # Load ground truth
            gt = current_ds.gt()
            # add a fake distance transform
            # we need this to safely replace this with the actual distance transforms later
            current_ds.add_input_from_data(np.zeros_like(gt, dtype = 'float32'))

            # Initialize correspondence list which makes sure that the same merge is not extracted from
            # multiple mc segmentations
            if ExperimentSettings().paths_avoid_duplicates:
                correspondence_list = []
            else:
                correspondence_list = None

            # loop over the different beta segmentations per train set
            for seg_id, seg_path in enumerate(paths_to_betas):
                key = keys_to_betas[seg_id]

                # Calculate the new distance transform and replace it in the dataset inputs
                seg = remove_small_segments(vigra.readHDF5(seg_path, key))
                dt  = distance_transform(seg, [1., 1., ExperimentSettings().anisotropy_factor])
                # NOTE IMPORTANT: We assume that the distance transform always has the last inp_id and that a (dummy) dt was already added in the beginning
                current_ds.replace_inp_from_data(current_ds.n_inp - 1, dt, clear_cache = False)
                # we delete all filters based on the distance transform
                current_ds.clear_filters(current_ds.n_inp - 1)

                # Compute the paths
                paths, _, path_classes, correspondence_list = extract_paths_and_labels_from_segmentation(
                        current_ds,
                        seg,
                        seg_id,
                        gt,
                        correspondence_list,
                        paths_cache_folder)

                if paths.size:
                    # TODO: decide which filters and sigmas to use here (needs to be exposed first)
                    features_train.append(path_feature_aggregator(current_ds, paths))
                    labels_train.append(path_classes)

                else:
                    print "No paths found for seg_id = {}".format(seg_id)
                    continue

        features_train = np.concatenate(features_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        assert features_train.shape[0] == labels_train.shape[0]
        features_train = np.nan_to_num(features_train).astype('float32')

        rf = RandomForest(
                features_train,
                labels_train,
                ExperimentSettings().n_trees,
                ExperimentSettings().n_threads)

        # cache the rf if caching is enabled
        if rf_cache_folder is not None:
            rf.write(rf_save_path, 'rf')

    return rf


def compute_false_merges(
        trainsets, # list of datasets with training data
        ds_test, # one dataset -> predict the false merged objects
        mc_segs_train, # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
        mc_segs_train_keys,
        mc_seg_test,
        mc_seg_test_key,
        rf_cache_folder = None,
        test_paths_cache_folder = None,
        train_paths_cache_folder = None
):
    """
    Computes and returns false merge candidates
    :param ds_train: Array of datasets representing multiple source images; [N x 1]
        Has to contain:
        ds_train.inp(0) := raw image
        ds_train.inp(1) := probs image
        ds_train.gt()   := groundtruth
    :param ds_test:
        Has to contain:
        ds_test.inp(0) := raw image
        ds_test.inp(1) := probs image
    :param mc_segs_train: Multiple multicut segmentations on ds_train
        Different betas for each ds_train; [N x len(betas)]
    :param mc_seg_test: Multicut segmentation on ds_test (usually beta=0.5)
    :return:
    """

    assert len(trainsets) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"
    assert len(mc_segs_train_keys) == len(mc_segs_train), "we must have the same number of segmentation vectors as trainsets"

    rf = train_random_forest_for_merges(
        trainsets,
        mc_segs_train,
        mc_segs_train_keys,
        rf_cache_folder,
        train_paths_cache_folder
    )

    # load the segmentation, compute distance transform and add it to the test dataset
    seg = vigra.readHDF5(mc_seg_test, mc_seg_test_key)
    dt = distance_transform(seg, [1., 1., ExperimentSettings().anisotropy_factor])
    ds_test.add_input_from_data(dt)

    paths_test, paths_to_objs_test = extract_paths_from_segmentation(
        ds_test,
        mc_seg_test,
        mc_seg_test_key,
        test_paths_cache_folder
    )

    assert len(paths_test) == len(paths_to_objs_test)

    features_test = path_feature_aggregator(
            ds_test,
            paths_test)
    assert features_test.shape[0] == len(paths_test)
    features_test = np.nan_to_num(features_test)

    # Cache features for debugging TODO deactivated for now
    #if not os.path.exists(paths_save_folder + '../debug'):
    #    os.mkdir(paths_save_folder + '../debug')
    #with open(paths_save_folder + '../debug/features_test.pkl', mode='w') as f:
    #    pickle.dump(features_test, f)

    return paths_test, rf.predict_probabilities(features_test)[:,1], paths_to_objs_test


# We sample new lifted edges and save them if a cache folder is given
def sample_and_save_paths_from_lifted_edges(
        cache_folder,
        ds,
        seg,
        obj_id,
        uv_local,
        distance_transform,
        eccentricity_centers,
        reverse_mapping):

    if cache_folder is not None:
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        save_path = os.path.join(cache_folder, 'paths_from_lifted_ds_%s_obj_%i.h5' % (ds.ds_name,obj_id))
    else:
        paths_save_file = ''

    # check if the cache already exists
    if os.path.exists(save_path): # if True, load paths from file
        paths_obj = vigra.readHDF5(save_path, 'paths')
        # we need to reshape the paths again to revover the coordinates
        if paths_obj.size:
            # FIXME This is a workaround to create the same type of np array even when len==1
            # FIXME I fear a similar issue when all paths have the exact same length
            if len(paths_obj) == 1:
                paths_obj = [ path.reshape( (len(path)/3, 3) ) for path in paths_obj ]
                tmp = np.empty((1,), dtype=np.object)
                tmp[0] = paths_obj[0]
                paths_obj = tmp
            else:
                paths_obj = np.array( [ path.reshape( (len(path)/3, 3) ) for path in paths_obj ] )
        uv_ids_paths_min_nh = vigra.readHDF5(save_path, 'uv_ids')

    else: # if False, compute the paths

        # Sample uv pairs out of seg_ids (make sure to have a minimal graph dist.)
        # ------------------------------------------------------------------------
        # TODO: Alternatively sample until enough false merges are found
        uv_ids_paths_min_nh = compute_and_save_long_range_nh(
            uv_local,
            ExperimentSettings().min_nh_range,
            ExperimentSettings().max_sample_size
        )

        if uv_ids_paths_min_nh.any():
            uv_ids_paths_min_nh = np.sort(uv_ids_paths_min_nh, axis = 1)

            # -------------------------------------------------------------
            # Get the distance transform of the current object

            masked_disttransf = distance_transform.copy()
            masked_disttransf[seg != obj_id] = np.inf

            # If we have a reverse mapping, turn them to the original labels
            uv_ids_paths_min_nh = np.array(
                    [np.array([reverse_mapping[u] for u in uv]) for uv in uv_ids_paths_min_nh])

            # Extract the respective coordinates from ecc_centers_seg thus creating pairs of coordinates
            uv_ids_paths_min_nh_coords = [[eccentricity_centers[u] for u in uv] for uv in uv_ids_paths_min_nh]

            # Compute the shortest paths according to the pairs list
            paths_obj = shortest_paths(
                masked_disttransf,
                uv_ids_paths_min_nh_coords,
                ExperimentSettings().n_threads)
            keep_mask = np.array([isinstance(x, np.ndarray) for x in paths_obj], dtype = np.bool)
            # FIXME This is a workaround to create the same type of np array even when len==1
            # FIXME I fear a similar issue when all paths have the exact same length
            if len(paths_obj) == 1:
                tmp = np.empty((1,), dtype=np.object)
                tmp[0] = paths_obj[0]
                paths_obj = tmp[keep_mask]
            else:
                paths_obj = np.array(paths_obj)[keep_mask]
            uv_ids_paths_min_nh = uv_ids_paths_min_nh[keep_mask]

        else:
            paths_obj = np.array([])

        # cache the paths if we have caching activated
        if cache_folder is not None:
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            paths_save = np.array([pp.flatten() for pp in paths_obj])
            try:
                # need to write paths with vlen and flatten before writing to properly save this
                with h5py.File(save_path) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(paths_save[0].dtype))
                    f.create_dataset('paths', data = paths_save, dtype = dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(paths_save, save_path, 'paths')

            vigra.writeHDF5(uv_ids_paths_min_nh, save_path, 'uv_ids')

    return paths_obj, uv_ids_paths_min_nh


# combine sampled and extra paths
def combine_paths(
        paths_obj,
        extra_paths,
        uv_ids_paths_min_nh,
        seg,
        mapping = None):

    # find coordinates belonging to the extra paths
    extra_coords = [[tuple(p[0]), tuple(p[-1])] for p in extra_paths]

    # map coordinates to uv ids
    if mapping is None:
        extra_path_uvs = np.array([np.array(
            [seg[coord[0]],
             seg[coord[1]]]) for coord in extra_coords])
    else:
        extra_path_uvs = np.array([np.array(
            [mapping[seg[coord[0]]],
             mapping[seg[coord[1]]]]) for coord in extra_coords])
    extra_path_uvs = np.sort(extra_path_uvs, axis=1)

    # exclude paths with u == v
    different_uvs = extra_path_uvs[:,0] != extra_path_uvs[:,1]
    extra_path_uvs = extra_path_uvs[different_uvs,:]
    extra_paths = extra_paths[different_uvs]

    # concatenate exta paths and sampled paths (modulu duplicates)
    if uv_ids_paths_min_nh.any(): # only concatenate if we have sampled paths
        matches = find_matching_row_indices(uv_ids_paths_min_nh, extra_path_uvs)
        if matches.size: # if we have matching uv ids, exclude them from the extra paths before concatenating
            duplicate_mask = np.ones(len(extra_path_uvs), dtype = np.bool)
            duplicate_mask[matches[:,1]] = False
            extra_path_uvs = extra_path_uvs[duplicate_mask]
            extra_paths = extra_paths[duplicate_mask]
        return np.concatenate([paths_obj, extra_paths]), np.concatenate([uv_ids_paths_min_nh, extra_path_uvs], axis = 0)

    else:
        return extra_paths, extra_path_uvs


# resolve each potential false merge individually with lifted edges
def resolve_merges_with_lifted_edges(
        ds,
        seg_id,
        false_paths, # dict(merge_ids : false_paths)
        path_rf,
        mc_segmentation,
        mc_weights_all, # the precomputed mc-weights
        paths_cache_folder=None,
        lifted_weights_all=None # pre-computed lifted mc-weights
):
    assert isinstance(false_paths, dict)

    # NOTE: We assume that the dataset already has a distance transform added as last input
    # This should work out, because we have already detected false merge paths for this segmentation
    disttransf = ds.inp(ds.n_inp - 1)
    # Pre-processing of the distance transform
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path
    #    detection) should be at the center of the current process
    disttransf = np.amax(disttransf) - disttransf
    #
    # c) Increase the value difference between pixels near the boundaries and pixels
    #    central within the processes. This increases the likelihood of the paths to
    #    follow the center of processes, thus avoiding short-cuts
    disttransf = np.power(disttransf, ExperimentSettings().paths_penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume

    # I have moved this to the dataset to have it cached
    ecc_centers_seg = ds.eccentricity_centers(seg_id, True)

    # get local and lifted uv ids
    uv_ids = ds._adjacent_segments(seg_id)
    uv_ids_lifted = compute_and_save_lifted_nh(
        ds,
        seg_id,
        ExperimentSettings().lifted_neighborhood,
        False
    )

    # iterate over the obj-ids which have a potential false merge
    # for each, sample new lifted edges and resolve the obj individually
    resolved_objs = {}
    for merge_id in false_paths:

        mask = mc_segmentation == merge_id
        seg_ids = np.unique(seg[mask])

        # map the extracted seg_ids to consecutive labels
        seg_ids_local, _, mapping = vigra.analysis.relabelConsecutive(seg_ids, start_label=0, keep_zeros=False)
        # mapping = old to new,
        # reverse = new to old
        reverse_mapping = {val: key for key, val in mapping.iteritems()}

        # FIXME Is this correct?
        # mask the local uv ids in this object
        local_uv_mask = np.in1d(uv_ids, seg_ids)
        local_uv_mask = local_uv_mask.reshape(uv_ids.shape).all(axis = 1)

        # extract local uv ids and corresponding weights
        uv_local = uv_ids[local_uv_mask]
        mc_weights = mc_weights_all[local_uv_mask]
        # map the uv ids to local labeling
        uv_local = np.array([[mapping[u] for u in uv] for uv in uv_local])

        # mask the lifted uv ids in this object
        lifted_uv_mask = np.in1d(uv_ids_lifted, seg_ids)
        lifted_uv_mask = lifted_uv_mask.reshape(uv_ids_lifted.shape).all(axis = 1)

        # extract the lifted uv ids and corresponding weights
        uv_local_lifted = uv_ids_lifted[lifted_uv_mask]
        lifted_weights = lifted_weights_all[lifted_uv_mask]
        uv_local_lifted = np.array([[mapping[u] for u in uv] for uv in uv_local_lifted])

        # sample new paths corresponding to lifted edges with min graph distance
        paths_obj, uv_ids_paths_min_nh = sample_and_save_paths_from_lifted_edges(
                paths_cache_folder,
                ds,
                mc_segmentation,
                merge_id,
                uv_local,
                disttransf,
                ecc_centers_seg,
                reverse_mapping)

        # Map to local uvs
        uv_ids_paths_min_nh = np.array([[mapping[u] for u in uv] for uv in uv_ids_paths_min_nh])

        # add the paths that were initially classified
        paths_obj, uv_ids_paths_min_nh = combine_paths(
            paths_obj,
            np.array(false_paths[merge_id]), # <- initial paths
            uv_ids_paths_min_nh,
            seg,
            mapping)

        if not paths_obj.size:
            continue

        # Compute the path features
        features = path_feature_aggregator(ds, paths_obj)
        features = np.nan_to_num(features)

        # Cache features for debug purpose # TODO disabled for now
        #with open(export_paths_path + '../debug/features_resolve_{}.pkl'.format(merge_id), mode='w') as f:
        #    pickle.dump(features, f)

        # compute the lifted weights from rf probabilities
        # FIXME TODO - not caching this for now -> should not be performance relevant
        lifted_path_weights = path_rf.predict_probabilities(features)[:,1]

        # Class 1: contain a merge
        # Class 0: don't contain a merge

        # scale the probabilities
        p_min = 0.001
        p_max = 1. - p_min
        lifted_path_weights = (p_max - p_min) * lifted_path_weights + p_min

        # Transform probs to weights
        lifted_path_weights = np.log((1 - lifted_path_weights) / lifted_path_weights)

        # Weighting edges with their length for proper lifted to local scaling
        lifted_path_weights /= lifted_path_weights.shape[0] * ExperimentSettings().lifted_path_weights_factor
        lifted_weights /= lifted_weights.shape[0]
        mc_weights /= mc_weights.shape[0]

        # Concatenate all lifted weights and edges
        if lifted_weights.size: # only concatenate if we have lifted edges from sampling
            lifted_weights = np.concatenate(
                (lifted_path_weights, lifted_weights),
                axis=0
            )
            uv_ids_lifted_nh_total = np.concatenate(
                (uv_ids_paths_min_nh, uv_local_lifted),
                axis=0
            )
        else:
            lifted_weights = lifted_path_weights
            uv_ids_lifted_nh_total = uv_ids_paths_min_nh

        resolved_nodes, _, _ = optimize_lifted(
            uv_local,
            uv_ids_lifted_nh_total,
            mc_weights,
            lifted_weights
        )

        resolved_nodes, _, _ = vigra.analysis.relabelConsecutive(resolved_nodes, start_label = 0, keep_zeros = False)
        # project back to global node ids and save
        resolved_objs[merge_id] = {reverse_mapping[i] : node_res for i, node_res in enumerate(resolved_nodes)}

    return resolved_objs


def resolve_merges_with_lifted_edges_global(
        ds,
        seg_id,
        false_paths, # dict(merge_ids : false_paths)
        path_rf,
        mc_segmentation,
        mc_weights_all, # the precomputed mc-weights
        paths_cache_folder = None,
        lifted_weights_all = None # pre-computed lifted mc-weights
):
    assert isinstance(false_paths, dict)

    # NOTE: We assume that the dataset already has a distance transform added as last input
    # This should work out, because we have already detected false merge paths for this segmentation
    disttransf = ds.inp(ds.n_inp - 1)
    # Pre-processing of the distance transform
    # a) Invert: the lowest values (i.e. the lowest penalty for the shortest path
    #    detection) should be at the center of the current process
    disttransf = np.amax(disttransf) - disttransf
    #
    # c) Increase the value difference between pixels near the boundaries and pixels
    #    central within the processes. This increases the likelihood of the paths to
    #    follow the center of processes, thus avoiding short-cuts
    disttransf = np.power(disttransf, ExperimentSettings().paths_penalty_power)

    # get the over-segmentation and get fragments corresponding to merge_id
    seg = ds.seg(seg_id)  # returns the over-segmentation as 3d volume

    # I have moved this to the dataset to have it cached
    ecc_centers_seg = ds.eccentricity_centers(seg_id, True)

    # get local and lifted uv ids
    uv_ids = ds._adjacent_segments(seg_id)
    uv_ids_lifted = compute_and_save_lifted_nh(
        ds,
        seg_id,
        ExperimentSettings().lifted_neighborhood,
        False
    )

    lifted_path_weights_all = []
    uv_ids_paths_min_nh_all = []

    for merge_id in false_paths:

        mask = mc_segmentation == merge_id
        seg_ids = np.unique(seg[mask])

        # extract the uv ids in this object
        local_uv_mask = np.in1d(uv_ids, seg_ids)
        local_uv_mask = local_uv_mask.reshape(uv_ids.shape).all(axis = 1)
        uv_ids_in_obj = uv_ids[local_uv_mask]

        # map the extracted seg_ids to consecutive labels
        seg_ids_local, _, mapping = vigra.analysis.relabelConsecutive(seg_ids, start_label=0, keep_zeros=False)
        reverse_mapping = {val: key for key, val in mapping.iteritems()}

        uv_ids_in_obj_local = np.array([[mapping[u] for u in uv] for uv in uv_ids_in_obj])

        # sample new paths corresponding to lifted edges with min graph distance
        paths_obj, uv_ids_paths_min_nh = sample_and_save_paths_from_lifted_edges(
            paths_cache_folder,
            ds,
            mc_segmentation,
            merge_id,
            uv_ids_in_obj_local,
            disttransf,
            ecc_centers_seg,
            reverse_mapping=reverse_mapping
        )

        # add the paths that were initially classified
        paths_obj, uv_ids_paths_min_nh = combine_paths(
            paths_obj,
            np.array(false_paths[merge_id]), # <- initial paths
            uv_ids_paths_min_nh,
            seg)

        if not paths_obj.size:
            continue

        # Compute the path features
        features = path_feature_aggregator(ds, paths_obj)
        features = np.nan_to_num(features)

        # Cache features for debug purpose # TODO not caching for now
        #with open(export_paths_path + '../debug/features_resolve_{}.pkl'.format(merge_id), mode='w') as f:
        #    pickle.dump(features, f)

        # compute the lifted weights from rf probabilities
        lifted_path_weights = path_rf.predict_probabilities(features)[:,1]

        # Class 1: contain a merge
        # Class 0: don't contain a merge

        # scale the probabilities
        p_min = 0.001
        p_max = 1. - p_min
        lifted_path_weights = (p_max - p_min) * lifted_path_weights + p_min

        # Transform probs to weights
        lifted_path_weights = np.log((1 - lifted_path_weights) / lifted_path_weights)

        lifted_path_weights_all.append(lifted_path_weights)
        uv_ids_paths_min_nh_all.append(uv_ids_paths_min_nh)

    lifted_path_weights_all = np.concatenate(lifted_path_weights_all)
    uv_ids_paths_min_nh_all = np.concatenate(uv_ids_paths_min_nh_all)

     # Weighting edges with their length for proper lifted to local scaling
    lifted_path_weights_all /= lifted_path_weights_all.shape[0] * ExperimentSettings().lifted_path_weights_factor
    lifted_weights_all /= lifted_weights_all.shape[0]
    mc_weights_all /= mc_weights_all.shape[0]

    # Concatenate all lifted weights and edges
    lifted_weights = np.concatenate(
        [lifted_path_weights_all, lifted_weights_all],
        axis=0
    )
    all_uv_ids_lifted_nh_total = np.concatenate(
        [uv_ids_paths_min_nh_all, uv_ids_lifted],
        axis=0
    )

    resolved_nodes, _, _ = optimize_lifted(
        uv_ids,
        all_uv_ids_lifted_nh_total,
        mc_weights_all,
        lifted_weights
    )

    resolved_nodes, _, _ = vigra.analysis.relabelConsecutive(resolved_nodes, start_label=0, keep_zeros=False)
    assert len(resolved_nodes) == uv_ids.max() + 1
    return resolved_nodes


def project_resolved_objects_to_segmentation(ds,
        seg_id,
        mc_segmentation,
        resolved_objs):

    rag = ds._rag(seg_id)
    mc_labeling, _ = rag.projectBaseGraphGt( mc_segmentation )
    new_label_offset = np.max(mc_labeling) + 1
    for obj in resolved_objs:
        resolved_nodes = resolved_objs[obj]
        for node_id in resolved_nodes:
            mc_labeling[node_id] = new_label_offset + resolved_nodes[node_id]
        new_label_offset += np.max(resolved_nodes.values()) + 1
    return rag.projectLabelsToBaseGraph(mc_labeling)
