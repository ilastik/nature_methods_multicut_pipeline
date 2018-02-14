import nifty_with_cplex as nifty
from math import sqrt
from Queue import LifoQueue, Queue
import numpy as np
from copy import deepcopy,copy
from skimage.morphology import skeletonize_3d
from ..ExperimentSettings import ExperimentSettings
from skimage.measure import label
from compute_paths_and_features import distance_transform

def close_cavities(volume):
    test_vol=np.zeros(volume.shape)
    test_vol[volume==0]=2
    test_vol[volume==1]=1
    test_vol=np.pad(test_vol, 1, "constant", constant_values=2)

    lab=label(test_vol,connectivity=1)
    if len(np.unique(lab))==2:
        del test_vol
        return volume, False
    count,what=0,0
    for uniq in np.unique(lab):
        if len(np.where(lab == uniq)[0])> count:
            count=len(np.where(lab == uniq)[0])
            what=uniq

    test_vol[lab==what]=0
    test_vol[lab != what] = 1
    del volume
    return test_vol[1:-1,1:-1,1:-1],True

def norm3d(point1,point2):

    anisotropy = [ExperimentSettings().anisotropy_factor, 1, 1]
    return sqrt(((point1[0] - point2[0])* anisotropy[0])*((point1[0] - point2[0])* anisotropy[0])+
                 ((point1[1] - point2[1])*anisotropy[1])*((point1[1] - point2[1])*anisotropy[1])+
                 ((point1[2] - point2[2]) * anisotropy[2])*((point1[2] - point2[2]) * anisotropy[2]))


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
    """searches for the first node to start with, explicitly searches preferably a node in the middle of the skeleton"""

    where = np.where(volume)

    if len(where[0]) == 0:
        return np.array([-1, -1, -1])
    point = np.array((where[0][0], where[1][0], where[2][0]))


    # point = np.array((where[:][0][0], where[:][1][0], where[:][2][0]))

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


def stage_one(skel_img, dt, anisotropy):
    """stage one, finds all nodes and edges, except for loops"""

    # initializing
    volume = deepcopy(skel_img)
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
        return is_node_map, is_term_map, is_branch_map, nodes, edges, loop_list

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued, is_node_list, are_near = check_box(volume, point, is_queued_map, is_node_map)
    nodes[current_node] = point

    while len(not_queued) == 0:
        volume[point[0], point[1], point[2]] = 0
        is_queued_map[point[0], point[1], point[2]] = 0
        nodes = {}
        point = init(volume)
        if (point == np.array([-1, -1, -1])).all():
            return is_node_map, is_term_map, is_branch_map, nodes, edges,loop_list
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

            length = length + norm3d(point,not_queued[0])
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
            edges.extend([[np.array([current_node, last_node]), length, edge_list, dt_list]])
            node_list.extend([[point[0], point[1], point[2]]])
            # putting node branches in the queue
            for x in not_queued:

                length = norm3d(point,x)
                queue.put(np.array([x, last_node, length,
                                    [[point[0], point[1], point[2]]],
                                    [dt[point[0], point[1], point[2]]]]))
                is_queued_map[x[0], x[1], x[2]] = 1

            is_branch_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node


    return is_node_map, is_term_map, is_branch_map, nodes, edges, loop_list


def stage_two(is_node_map, list_term, edges, dt):
    """finds edges for loops"""
    i=0


    for point in list_term:
        _, _, list_near_nodes = check_box(is_node_map, point, np.zeros(is_node_map.shape, dtype=int),
                                          np.zeros(is_node_map.shape, dtype=int), 2)

        if len(list_near_nodes) != 0:
            i=i+1

    assert (i < 2)



def form_term_list_with_cents(is_term_map, border_points, mode):
    """returns list of terminal points taken from an image"""

    border_distance=ExperimentSettings().border_distance
    term_where = np.array(np.where(is_term_map)).transpose()

    dict_border_points = {key: [] for key in xrange(0, len(border_points))}

    if mode=="testing":

        for idx, cent_point in enumerate(border_points):
            comparison_array = []

            [comparison_array.append(norm3d(term_point, cent_point))
             for term_point in term_where]

            min_index = np.argmin(comparison_array)

            dict_border_points[idx]=[
                    is_term_map[term_where[min_index][0],
                                term_where[min_index][1],
                                term_where[min_index][2]]-1]

            term_where = np.delete(term_where, min_index, axis=0)


    else:

        for idx, cent_point in enumerate(border_points):
            comparison_array = []

            [comparison_array.append(norm3d(term_point, cent_point))
             for term_point in term_where]

            min_index = np.argmin(comparison_array)

            if comparison_array[min_index] <= border_distance:
                dict_border_points[idx]=[is_term_map[term_where[index][0],
                                term_where[index][1],
                                term_where[index][2]]-1
                                         for index, val in enumerate(comparison_array)
                                         if val<=border_distance]


            else:
                dict_border_points[idx]=[is_term_map[term_where[min_index][0],
                                term_where[min_index][1],
                                term_where[min_index][2]]-1]



    return dict_border_points


def form_term_list(is_term_map):
    """returns list of terminal points taken from an image"""


    term_where = np.array(np.where(is_term_map)).transpose()

    term_list = []
    for point in term_where:
        term_list.extend([is_term_map[point[0], point[1], point[2]]])
    term_list = np.array([term for term in term_list])

    return term_list


def skeleton_to_graph(skel_img, dt, anisotropy):
    """main function, wraps up stage one and two"""

    is_node_map, is_term_map, is_branch_map, nodes, edges_and_lens, loop_list = \
        stage_one(skel_img, dt, anisotropy)



    if len(nodes) < 2:
        return nodes, np.array(edges_and_lens), [], loop_list


    # stage_two(is_node_map, list_term_unfinished, edges_and_lens, dt)

    edges_and_lens = [[val1, val2, val3, max(val4)] for val1, val2, val3, val4 in edges_and_lens]

    # term_list -= 1



    # loop_list -= 1
    return nodes, np.array(edges_and_lens), is_term_map, loop_list


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



def terminal_func(start_queue,g,finished_dict,node_dict,main_dict,edges,nodes_list,intersecting_node_dict):

    queue = Queue()
    queued_list=[]

    while start_queue.qsize():

        # draw from queue
        current_node, label = start_queue.get()


        #check the adjacency
        adjacency = np.array([[adj_node, adj_edge] for adj_node, adj_edge
                              in g.nodeAdjacency(current_node)])




        assert(len(adjacency) == 1)

        # for terminating points
        if len(adjacency) == 1:

            if (edges[adjacency[0][1]][2][0]==np.array(nodes_list[current_node+1])).all():

                main_dict[current_node] = [[current_node, adjacency[0][0]],
                                           edges[adjacency[0][1]][1],
                                           edges[adjacency[0][1]][2],
                                           adjacency[0][1],
                                           edges[adjacency[0][1]][3]]


            else:

                main_dict[current_node] = [[current_node, adjacency[0][0]],
                                           edges[adjacency[0][1]][1],
                                           edges[adjacency[0][1]][2][::-1],
                                           adjacency[0][1],
                                           edges[adjacency[0][1]][3]]


            # if adjacent node was already visited
            if adjacency[0][0] in node_dict.keys():

                node_dict[adjacency[0][0]][0].remove(current_node)
                node_dict[adjacency[0][0]][2].remove(adjacency[0][1])

                # if this edge is longer than already written edge
                if (edges[adjacency[0][1]][1]) >= \
                        (main_dict[node_dict[adjacency[0][0]][1]][1]):
                    #writing to intersection_node so we can connect later on
                    if adjacency[0][0] in intersecting_node_dict.keys():
                        intersecting_node_dict[adjacency[0][0]].append(node_dict[adjacency[0][0]][1])
                    else:
                        intersecting_node_dict[adjacency[0][0]]=[node_dict[adjacency[0][0]][1]]

                    finished_dict[node_dict[adjacency[0][0]][1]] \
                        = deepcopy(main_dict[node_dict[adjacency[0][0]][1]])

                    del main_dict[node_dict[adjacency[0][0]][1]]
                    node_dict[adjacency[0][0]][1] = current_node

                else:
                    #writing to intersection_node so we can connect later on
                    if adjacency[0][0] in intersecting_node_dict.keys():
                        intersecting_node_dict[adjacency[0][0]].append(current_node)
                    else:
                        intersecting_node_dict[adjacency[0][0]]=[current_node]

                    finished_dict[current_node] = deepcopy(main_dict[current_node])

                    # get unique rows
                    # finished_dict[current_node][2]=\
                    #     get_unique_rows(np.array(finished_dict[current_node][2]))
                    del main_dict[current_node]

            # create new dict.key for adjacent node
            else:

                node_dict[adjacency[0][0]] = [[adj_node for adj_node, adj_edge
                                               in g.nodeAdjacency(adjacency[0][0])
                                               if adj_node != current_node],
                                              current_node,
                                              [adj_edge for adj_node, adj_edge
                                               in g.nodeAdjacency(adjacency[0][0])
                                               if adj_edge != adjacency[0][1]]]



            # we finish when we have only two open labels left
            if start_queue.qsize() == 0 and len(nodes_list.keys())==6:

                # writing longest label at node to the others,
                # so we dont cut in half later
                for finished_label in [adj_node for adj_node,
                                                    adj_edge in g.nodeAdjacency(adjacency[0][0])
                                       if adj_node != node_dict[adjacency[0][0]][0][0]
                                       and adj_node != node_dict[adjacency[0][0]][1]]:
                    finished_dict[finished_label]. \
                        append(finished_label)

                for key in main_dict.keys():

                    # writing to intersection_node so we can connect later on
                    if main_dict[key][0][-1] in intersecting_node_dict.keys():
                        intersecting_node_dict[main_dict[key][0][-1]].append(key)
                    else:
                        intersecting_node_dict[main_dict[key][0][-1]] = [key]

                    finished_dict[key] = deepcopy(main_dict[key])

                    del main_dict[key]

                    finished_dict[key].append(key)

                # deleting node from dict
                # del node_dict[current_node]
                _ = queue.get()
                break

            #spooky things where we had loops or crosses
            if len(edges)==4 and len(nodes_list)==5 and start_queue.qsize()==0:
                for dummy_lbl in main_dict.keys():
                    main_dict[dummy_lbl].append(dummy_lbl)
                    finished_dict[dummy_lbl] = deepcopy(main_dict[dummy_lbl])

                    del main_dict[dummy_lbl]
                while queue.qsize()!=0:
                    _ = queue.get()
                break

            # if all except one branches reached the adjacent node
            if len(node_dict[adjacency[0][0]][0]) == 1:


                # writing longest label at node to the others,
                # so we dont cut in half later
                # noblankspace
                for finished_label in [adj_node for adj_node,
                                           adj_edge in g.nodeAdjacency(adjacency[0][0])
                              if adj_node!=node_dict[adjacency[0][0]][0][0]
                              and adj_node!=node_dict[adjacency[0][0]][1]]:

                    finished_dict[finished_label].\
                        append(node_dict[adjacency[0][0]][1])




                # writing new node to label
                main_dict[node_dict[adjacency[0][0]][1]][0].\
                    extend([node_dict[adjacency[0][0]][0][0]])

                #comparing maximum of dt
                if main_dict[node_dict[adjacency[0][0]][1]][4] <\
                        edges[node_dict[adjacency[0][0]][2][0]][3]:

                    main_dict[node_dict[adjacency[0][0]][1]][4]=\
                        edges[node_dict[adjacency[0][0]][2][0]][3]

                # adding length to label
                main_dict[node_dict[adjacency[0][0]][1]][1] += \
                    edges[node_dict[adjacency[0][0]][2][0]][1]

                # adding path to next node to label
                if main_dict[node_dict[adjacency[0][0]][1]][2][-1]==\
                        edges[node_dict[adjacency[0][0]][2][0]][2][-1]:

                    main_dict[node_dict[adjacency[0][0]][1]][2].extend(
                        edges[node_dict[adjacency[0][0]][2][0]][2][-2::-1])

                # adding path to next node to label
                else:

                    main_dict[node_dict[adjacency[0][0]][1]][2].extend(
                        edges[node_dict[adjacency[0][0]][2][0]][2][1:])

                #adding edge number to label
                main_dict[node_dict[adjacency[0][0]][1]][3]=\
                    node_dict[adjacency[0][0]][2][0]

                # putting next
                queue.put([node_dict[adjacency[0][0]][0][0],
                           node_dict[adjacency[0][0]][1]])
                queued_list.append(node_dict[adjacency[0][0]][1])


                # # deleting node from dict
                # del node_dict[adjacency[0][0]]

    if (len(main_dict.keys()))==1 and len(nodes_list.keys())==4:
        key=main_dict.keys()[0]
        # writing to intersection_node so we can connect later on
        if main_dict[key][0][-2] in intersecting_node_dict.keys():
            intersecting_node_dict[main_dict[key][0][-2]].append(key)
        else:
            intersecting_node_dict[main_dict[key][0][-2]] = [key]

        for possible_unfinished_key in finished_dict.keys():
            if len(finished_dict[possible_unfinished_key])<6:
                finished_dict[possible_unfinished_key].append(main_dict.keys()[0])

        for key in main_dict.keys():
            finished_dict[key] = deepcopy(main_dict[key])

            del main_dict[key]

            finished_dict[key].append(key)
        _ = queue.get()

    return queue,finished_dict,node_dict,main_dict,intersecting_node_dict,queued_list


def pruning_2_nodes(edges,term_list):

    assert len(edges)==1, "if we have two nodes we should have only one edge"
    assert len(term_list)==2,"if we have two nodes we should have two term points"

    ratio=(edges[0][1]/2)/edges[0][3]

    if ratio>ExperimentSettings().pruning_factor:
        return np.array(term_list)
    else:
        return []





#TODO check whether edgelist and termlist is ok (because of -1)
def graph_pruning(g,term_list,edges,nodes_list, dict_border_points=None):

    finished_dict={}
    node_dict={}
    main_dict={}
    start_queue=Queue()
    intersecting_node_dict={}
    last_dict={}
    # edges_and_lens=deepcopy(edges)
    last_point_was_a_loop=0


    for term_point in term_list:
        start_queue.put([term_point,term_point])

    assert start_queue.qsize()!=1
    #TODO implement clean case for 3 or 2 term_points
    if start_queue.qsize()==2:
        return pruning_2_nodes(edges,term_list)

    queue,finished_dict,node_dict,main_dict,intersecting_node_dict,queued_list = \
        terminal_func (start_queue, g, finished_dict,
                       node_dict, main_dict, copy(edges),
                       nodes_list,intersecting_node_dict)

    assert queue.qsize()!=1

    while queue.qsize():


        # draw from queue
        current_node, label = queue.get()

        min_length_array=[main_dict[key][1] for key in queued_list]

        # if this path is not the smallest right now, skip
        if main_dict[label][1] != min(min_length_array):
            queue.put([current_node, label])
            continue

        queued_list.remove(label)


        # if current node was already visited at least once
        if current_node in node_dict.keys() and \
                main_dict[label][0][-2] in node_dict[current_node][0]:



            # remove previous node from adjacency
            node_dict[current_node][0].remove(main_dict[label][0][-2])

            # remove previous edge from adjacency
            node_dict[current_node][2].remove(main_dict[label][3])


            # if current label is longer than longest in node
            if (main_dict[label][1]) >= \
                    (main_dict[node_dict[current_node][1]][1]):

                # writing to intersection_node so we can connect later on
                if current_node in intersecting_node_dict.keys():
                    intersecting_node_dict[current_node].append(node_dict[current_node][1])
                else:
                    intersecting_node_dict[current_node] = [node_dict[current_node][1]]


                # writing winner node in loser node
                #noblankspace
                main_dict[node_dict[current_node][1]].append(label)

                # finishing previous longest label
                finished_dict[node_dict[current_node][1]] \
                    = deepcopy(main_dict[node_dict[current_node][1]])
                del main_dict[node_dict[current_node][1]]

                # get unique rows
                # finished_dict[node_dict[current_node][1]][2]\
                #     =get_unique_rows(np.array
                #                      (finished_dict[node_dict[current_node][1]][2]))

                # writing new label to longest in node
                node_dict[current_node][1]=label


            else:

                if current_node in intersecting_node_dict.keys():
                    intersecting_node_dict[current_node].append(label)
                else:
                    intersecting_node_dict[current_node] = [label]


                #finishing this label
                finished_dict[label] = deepcopy(main_dict[label])

                # get unique rows
                # finished_dict[label][2]=\
                #     get_unique_rows(np.array(finished_dict[label][2]))

                del main_dict[label]



        elif current_node not in node_dict.keys():

            #create new entry for this node
            node_dict[current_node] = [[adj_node for adj_node, adj_edge
                                           in g.nodeAdjacency(current_node)
                                           if adj_node != main_dict[label][0][-2]],
                                          label,
                                          [adj_edge for adj_node, adj_edge
                                           in g.nodeAdjacency(current_node)
                                           if adj_edge != main_dict[label][3]]]


        #finishing contraction
        if len(main_dict.keys())<3 :

            finished_label_arr=[]
            left_keys=main_dict.keys()

            assert main_dict[left_keys[0]][0][-2] not in main_dict[left_keys[1]][0], "last labels share an edge"
            assert main_dict[left_keys[1]][0][-2] not in main_dict[left_keys[0]][0], "last labels share an edge"


            for left_key in left_keys:

                last_node = main_dict[left_key][0][-1]
                pre_last_node=main_dict[left_key][0][-2]
                if last_node not in node_dict.keys():
                    last_node = main_dict[left_key][0][-2]
                    pre_last_node = main_dict[left_key][0][-3]
                next_node = node_dict[last_node][0][0]
                adj_nodes=np.array([[adj_node,adj_edge] for adj_node, adj_edge in g.nodeAdjacency(last_node) if adj_node != pre_last_node])
                if len(adj_nodes) > 1:
                    finished_label_arr.append([adj_node[0] for adj_node in adj_nodes if adj_node[0] not in [next_node, main_dict[left_keys[0]][0][-2],main_dict[left_keys[1]][0][-2]] ])
                else:
                    finished_label_arr.append(np.array([]))

            for idx,finished_labels in enumerate(finished_label_arr):

                if len(finished_labels) != 0:

                    for finished_label in finished_labels:

                        if finished_label in finished_dict.keys():

                            finished_dict[finished_label] \
                                .append(left_keys[idx])

                        else:

                            finished_dict[node_dict[finished_label][1]] \
                                .append(left_keys[idx])

            while main_dict[left_keys[0]][0][-1]!=main_dict[left_keys[1]][0][-1]:
                short_key=left_keys[np.argmin([main_dict[left_keys[0]][1],main_dict[left_keys[1]][1]])]
                last_node=main_dict[short_key][0][-1]
                pre_last_node=main_dict[short_key][0][-2]

                adj_nodes=np.array([[adj_node,adj_edge] for adj_node, adj_edge in g.nodeAdjacency(last_node) if adj_node != pre_last_node])
                assert len(adj_nodes)>0
                if len(adj_nodes)>1:
                    next_node=node_dict[last_node][0][0]
                    next_edge = node_dict[last_node][2][0]
                    finished_label_arr=adj_nodes[adj_nodes[:,0]!=next_node][:,0]
                else:
                    next_node = adj_nodes[0][0]
                    next_edge = adj_nodes[0][1]
                    finished_label_arr=None


                # writing new node to label
                main_dict[short_key][0]. \
                    extend([next_node])

                # comparing maximum of dt
                if main_dict[short_key][4] < \
                        edges[next_edge][3]:
                    main_dict[short_key][4] = \
                        edges[next_edge][3]

                    # adding length to label
                    main_dict[short_key][1] += \
                        edges[next_edge][1]

                # adding path to next node to label
                if main_dict[short_key][2][-1] == \
                        edges[next_edge][2][-1]:

                    main_dict[short_key][2].extend(
                        edges[next_edge][2][-2::-1])

                # adding path to next node to label
                else:

                    main_dict[short_key][2].extend(
                        edges[next_edge][2][1:])

                # adding edge number to label
                main_dict[short_key][3] = \
                    next_edge




            assert main_dict[main_dict.keys()[0]][0][-1]==main_dict[main_dict.keys()[1]][0][-1], "something went wrong with the new stuff"

            for key in main_dict.keys():

                # writing to intersection_node so we can connect later on
                if main_dict[key][0][-1] in intersecting_node_dict.keys():
                    if key not in intersecting_node_dict[main_dict[key][0][-1]]:
                        intersecting_node_dict[main_dict[key][0][-1]].append(key)
                else:
                    intersecting_node_dict[main_dict[key][0][-1]] = [key]

                finished_dict[key]=deepcopy(main_dict[key])

                del main_dict[key]

                finished_dict[key].append(key)

            # # deleting node from dict
            # del node_dict[current_node]

            break

        # check if all branches reached the adjacent node but we cant go
        # further because the dominant edge is not the smallest edge
        if main_dict[node_dict[current_node][1]][1] > min(min_length_array) and len(node_dict[current_node][0]) == 1:
            queue.put([current_node,node_dict[current_node][1]])
            queued_list.append(node_dict[current_node][1])


        # if all except one branches reached the adjacent node
        elif len(node_dict[current_node][0]) == 1:

            # writing longest label at node to the others,
            # so we dont cut in half later
            for finished_label in [adj_node for adj_node,
                                                adj_edge in g.nodeAdjacency(current_node)
                                   if adj_node!=node_dict[current_node][0][0] and
                                                   adj_node!=main_dict[node_dict[current_node][1]][0][-2]]:

                if finished_label in finished_dict.keys():

                    finished_dict[finished_label] \
                        .append(node_dict[current_node][1])

                else:

                    finished_dict[node_dict[finished_label][1]]\
                        .append(node_dict[current_node][1])



            # writing new node to label
            main_dict[node_dict[current_node][1]][0]. \
                extend([node_dict[current_node][0][0]])

            # comparing maximum of dt
            if main_dict[node_dict[current_node][1]][4] < \
                    edges[node_dict[current_node][2][0]][3]:

                main_dict[node_dict[current_node][1]][4]=\
                    edges[node_dict[current_node][2][0]][3]

            # adding length to label
            main_dict[node_dict[current_node][1]][1] += \
                edges[node_dict[current_node][2][0]][1]

            # adding path to next node to label
            if main_dict[node_dict[current_node][1]][2][-1]==\
                    edges[node_dict[current_node][2][0]][2][-1]:

                main_dict[node_dict[current_node][1]][2].extend(
                    edges[node_dict[current_node][2][0]][2][-2::-1])

            # adding path to next node to label
            else:

                main_dict[node_dict[current_node][1]][2].extend(
                    edges[node_dict[current_node][2][0]][2][1:])

            # adding edge number to label
            main_dict[node_dict[current_node][1]][3] = \
                node_dict[current_node][2][0]

            # putting next
            queue.put([node_dict[current_node][0][0],
                        node_dict[current_node][1]])
            queued_list.append(node_dict[current_node][1])

            # # deleting node from dict
            # del node_dict[current_node]


        assert(queue.qsize()>0),"contraction finished before all the nodes were seen"



    #Here begins the actual pruning
    # TODO maybe faster ?
    pre_plotting_dict = \
        {key: deepcopy(finished_dict[key][5]) for key in finished_dict.keys()
         if finished_dict[key][1] / finished_dict[key][4] > ExperimentSettings().pruning_factor}




    counter = 1
    while counter != 0:
        counter = 0
        for key in pre_plotting_dict.keys():
            if pre_plotting_dict[key] not in pre_plotting_dict.keys():
                counter = 1

                del pre_plotting_dict[key]

    pruned_term_list = [key for key in pre_plotting_dict.keys()]




    if len(pruned_term_list)==1:

        key=pruned_term_list[0]

        if len(nodes_list.keys())==4:
            node = finished_dict[key][0][-2]
        else:

            node=finished_dict[key][0][-1]
        list=np.array(intersecting_node_dict[node])[np.array(intersecting_node_dict[node])!=key]
        count=[0,0]
        for possible_lbl in list:
            possible_ratio=(finished_dict[possible_lbl][1]+finished_dict[key][1])/\
                           max(finished_dict[possible_lbl][4],finished_dict[key][4])
            if possible_ratio>count[1]:
                count[0]=possible_lbl
                count[1]=possible_ratio

        pruned_term_list.append(count[0])

    return np.array(pruned_term_list)


def edge_paths_and_counts_for_nodes(g, weights, dict_border_points,
                                    pruned_term_list, mode, n_threads=8):
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

    max_number_of_paths=ExperimentSettings().max_number_of_paths_for_training

    edge_paths = {}
    edge_counts = np.zeros(g.numberOfEdges, dtype='uint32')

    # build the nifty shortest path object
    path_finder = nifty.graph.ShortestPathDijkstra(g)

    # if we have at least one central point
    if dict_border_points is not None:

        dict_border_keys=dict_border_points.keys()


        if len(dict_border_keys)>1:

            for idx,key in enumerate(dict_border_keys[:-1]):

                target_nodes=np.concatenate([dict_border_points[left_key]
                                             for left_key in dict_border_keys[idx + 1:]])


                # iterate over the source nodes
                # we don't need to go to the last node, because it won't have any more targets
                for ii, u in enumerate(dict_border_points[key]):

                    # find the shortest path from source node u to the target nodes
                    shortest_paths = path_finder.runSingleSourceMultiTarget(
                            weights.tolist(),
                            int(u),
                            np.uint32(target_nodes),
                            returnNodes=False
                    )
                    assert len(shortest_paths) == len(target_nodes)

                    # extract the shortest path for each node pair and
                    # increase the edge counts
                    for jj, sp in enumerate(shortest_paths):
                        v = target_nodes[jj]
                        edge_paths[(u, v)] = sp
                        edge_counts[sp] += 1

            edge_paths_finished=edge_paths

        # if we only have one central point
        else :

            target_nodes=pruned_term_list

            #deleting the actual points from term_list if they are there
            for single_point in dict_border_points[0]:
                    if single_point in target_nodes:
                        target_nodes=np.delete(
                            target_nodes, np.where(target_nodes==single_point)[0][0])


            for ii, u in enumerate(dict_border_points[0]):

                # find the shortest path from source node u to the target nodes
                shortest_paths = path_finder.runSingleSourceMultiTarget(
                    weights.tolist(),
                    int(u),
                    np.uint32(target_nodes),
                    returnNodes=False
                )
                assert len(shortest_paths) == len(target_nodes)

                # extract the shortest path for each node pair and
                # increase the edge counts
                for jj, sp in enumerate(shortest_paths):
                    v = target_nodes[jj]
                    edge_paths[(u, v)] = sp
                    edge_counts[sp] += 1

            edge_paths_keys=edge_paths.keys()

            max_sum=0
            for key in edge_paths_keys:
                sum_weights = sum(weights[edge_paths[key]])
                if sum_weights>max_sum:
                    max_sum=sum_weights
                    counter = key


            edge_paths_finished={}
            edge_paths_finished[counter]=edge_paths[counter]



    # if we have no central points: build paths from pruning
    else :
        if len(pruned_term_list)<2:
            return None
        assert (mode != "testing"), "we want more than one path for testing"
        # only build one path
        if mode == "testing":

            for ii, u in enumerate(pruned_term_list[:-1]):

                target_nodes = pruned_term_list[ii + 1:]

                shortest_paths = path_finder.runSingleSourceMultiTarget(
                    weights.tolist(),
                    u,
                    target_nodes,
                    returnNodes=False
                )
                assert len(shortest_paths) == len(target_nodes)

                for jj, sp in enumerate(shortest_paths):
                    v = target_nodes[jj]
                    edge_paths[(u, v)] = sp
                    edge_counts[sp] += 1

            edge_paths_keys = edge_paths.keys()

            max_sum = 0
            for key in edge_paths_keys:
                sum_weights = sum(weights[edge_paths[key]])
                # print "key: ",key,", sum: ",sum_weights
                if sum_weights > max_sum:
                    max_sum = sum_weights
                    counter = key

            edge_paths_finished = {}
            edge_paths_finished[counter] = edge_paths[counter]

        # build many paths for training
        else:

            for ii, u in enumerate(pruned_term_list[:-1]):

                target_nodes = pruned_term_list[ii + 1:]

                shortest_paths = path_finder.runSingleSourceMultiTarget(
                    weights.tolist(),
                    u,
                    target_nodes,
                    returnNodes=False
                )
                assert len(shortest_paths) == len(target_nodes)

                for jj, sp in enumerate(shortest_paths):
                    v = target_nodes[jj]
                    edge_paths[(u, v)] = sp
                    edge_counts[sp] += 1

            edge_paths_keys = edge_paths.keys()

            edge_paths_finished = {key:edge_paths[key] for key in edge_paths_keys}



    return edge_paths_finished


def check_edge_paths_for_list(edge_paths, node_list):
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

    # print "passed"

# def check_edge_paths_for_dict(edge_paths, term_dict):
#     """checks edge paths (constantin)"""
#
#     from itertools import combinations
#     pairs = combinations(node_list, 2)
#     pair_list = [pair for pair in pairs]
#
#     # make sure that we have all combination in the edge_paths
#     for pair in pair_list:
#         assert pair in edge_paths
#
#     # make sure that we don't have any spurious pairs in edge_paths
#     for pair in edge_paths:
#         assert pair in pair_list
#
#     # print "passed"

def build_paths_from_edges(edge_paths,edges):
    """Builds paths from edges """

    finished_paths = []

    for pair in edge_paths.keys():

        single_path = []

        if len(edge_paths[pair]) > 1:

            if edges[edge_paths[pair][0]][2][0] == \
                    edges[edge_paths[pair][1]][2][0] or edges[edge_paths[pair][0]][2][0] == \
                    edges[edge_paths[pair][1]][2][-1]:

                single_path.extend(edges[edge_paths[pair][0]][2][::-1])

            else:
                single_path.extend(edges[edge_paths[pair][0]][2])

            for edge in edge_paths[pair][1:]:

                if edges[edge][2][0] == single_path[-1]:
                    single_path.extend(edges[edge][2][1:])

                elif edges[edge][2][-1] == single_path[-1]:
                    single_path.extend(edges[edge][2][-2::-1])

                else:
                    assert (1 == 2), (edge, edge_paths[pair])
        else:
            single_path.extend(edges[edge_paths[pair][0]][2])

        finished_paths.extend([single_path])

    return finished_paths


def compute_graph_and_paths(img, dt, anisotropy,
                            border_points, mode, dict_border_points=None):
    """ overall wrapper for all functions, input: label image; output: paths
        sampled from skeleton
    """
    #expand image as the skeletonization sees the image border as a border
    expand_number=ExperimentSettings().pad_width
    #skeletonize
    # skel_img=skeletonize_3d(np.pad(img, expand_number,"edge"))[expand_number:-expand_number,
    #                                                              expand_number:-expand_number,
    #                                                              expand_number:-expand_number]

    skel_img=skeletonize_3d(img)
    nodes, edges_and_lens, is_term_map, loop_list = \
        skeleton_to_graph(skel_img, dt, anisotropy)

    # print "deleting skel_img..."
    del skel_img


    if len(nodes) < 2:
        return []
    g, edge_lens, edges_and_lens = \
        graph_and_edge_weights(nodes, edges_and_lens)

    for_building=deepcopy(edges_and_lens)
    check_connected_components(g)



    if len(border_points)>1:
        dict_border_points = form_term_list_with_cents(is_term_map, border_points, mode)

        # making sure we can actually compute some paths
        for key in dict_border_points.keys():
            assert(len(dict_border_points[key])>0)
        pruned_term_list=None

    elif len(border_points) == 1 :

        term_list_before_copy=form_term_list(is_term_map)
        term_list=deepcopy(term_list_before_copy)
        term_list -= 1

        dict_border_points = form_term_list_with_cents(is_term_map, border_points, mode)

        # making sure we can actually compute some paths
        for key in dict_border_points.keys():
            assert(len(dict_border_points[key])>0)

        #FOR PRUNING
        ##########################################


        loop_uniq, loop_nr = np.unique(loop_list, return_counts=True)

        for where in np.where(loop_nr > 1)[0]:

            adjacency = np.array([[adj_node, adj_edge]
                                    for adj_node, adj_edge
                                    in g.nodeAdjacency(loop_uniq[where] - 1)])

            if (len(adjacency)) == 1:
                term_list = np.append(term_list, loop_uniq[where] - 1)

        pruned_term_list = graph_pruning\
                (g, term_list, edges_and_lens, nodes, dict_border_points)
            ##########################################

        if len(pruned_term_list)==0:
            return []
    else:

        term_list= form_term_list(is_term_map)
        term_list -= 1

        #FOR PRUNING
        ##########################################


        loop_uniq, loop_nr = np.unique(loop_list, return_counts=True)

        for where in np.where(loop_nr > 1)[0]:

            adjacency = np.array([[adj_node, adj_edge]
                                    for adj_node, adj_edge
                                    in g.nodeAdjacency(loop_uniq[where] - 1)])

            if (len(adjacency)) == 1:
                term_list = np.append(term_list, loop_uniq[where] - 1)

        pruned_term_list = graph_pruning\
                (g, term_list, edges_and_lens, nodes)
            ##########################################
        if len(pruned_term_list)==0:
            return []
        dict_border_points=None

    #TODO cores global
    edge_paths = \
        edge_paths_and_counts_for_nodes\
            (g,edge_lens,dict_border_points,pruned_term_list, "training", 1)

    if edge_paths==None:
        return []

    finished_paths=build_paths_from_edges(edge_paths,for_building)

    return finished_paths


def shorten_paths(paths_full_raw):
    """for shortening paths so they dont end exactly at the border """

    shortage_ratio=ExperimentSettings().ratio_for_shortage
    if shortage_ratio==0:
        return paths_full_raw

    assert(shortage_ratio<=0.5)

    paths_full=[path for path in paths_full_raw if len(path)>5]

    if len(paths_full)==0:
        return paths_full_raw

    # collects the "length" for every point of every path
    lens_paths = [np.concatenate(([norm3d(path[0], path[1]) / 2],
                                 [norm3d(path[idx - 1], path[idx]) / 2 +
                                  norm3d(path[idx], path[idx + 1]) / 2
                                  for idx, point in enumerate(path)
                                  if idx != 0 and idx != len(path) - 1],
                                 [norm3d(path[-2], path[-1]) / 2]))
                 for path in paths_full]

    overall_lengths=[sum(val) for val in lens_paths]

    shortage_lengths=[overall_length*shortage_ratio for overall_length in overall_lengths]

    indexes_front_back_array=[path_index_shortage_pointss_return(lens_paths[idx],shortage_lengths[idx]) for idx,val in enumerate(lens_paths)]

    paths_cut=[paths_full[idx][indexes_front_back_array[idx][0]:indexes_front_back_array[idx][1]] for idx,val in enumerate(paths_full)]

    return paths_cut


def path_index_shortage_pointss_return(lens_path,shortage_length):
    """returns index point where we cut off our paths for shortage"""


    sum_front=0
    sum_back=0


    for idx,val in enumerate(lens_path):

        sum_front=sum_front+val

        if sum_front >= shortage_length:
            index_front = idx
            break


    for idx, val in enumerate(lens_path[::-1]):

        sum_back = sum_back + val

        if sum_back >= shortage_length:
            index_back = len(lens_path)- 1 - idx
            break


    assert (index_front <= index_back)

    # assuring we dont
    if index_back-index_front<2:
        index_front=index_front-1
        index_back=index_back+1

    assert (index_back < len(lens_path))
    assert (index_front >= 0)

    return index_front,index_back





def parallel_wrapper(idx,seg, dt, gt, anisotropy,
                      label, len_uniq,
                     border_points=[], mode="training"):
    assert (mode=="testing" or mode=="training")

    if idx%50==0:
        print "Label nr. ", idx , " of ", len_uniq

    # masking volume
    img=np.zeros(seg.shape,dtype="uint8")
    img[seg==label]=1
    img, bool_closed=close_cavities(img)
    if bool_closed==True:
        dt=distance_transform(np.uint64(img),
                              [ExperimentSettings().anisotropy_factor,1.,1.])
    paths_full = compute_graph_and_paths(img, dt, anisotropy,
                                    border_points, mode)
    del dt
    del img
    del seg
    paths = shorten_paths(paths_full)
    del paths_full
    if len(paths) == 0:
        print "no paths found for label ", label



    if mode=="testing":

        if len(paths) == 0:
            return [],[]

        all_paths_single=[np.array(path) for path in paths]
        paths_to_objs_single=[label for path in paths]


        return all_paths_single, paths_to_objs_single

    else:


        if len(paths) == 0:
            return [],[],[]

        all_paths_single=[np.array(path) for path in paths]

        paths_to_objs_single = [label for x in xrange(0, len(paths))]

        # path_classes_single = [
        #     False if gt[path[0][0], path[0][1], path[0][2]] ==
        #             gt[path[-1][0], path[-1][1], path[-1][2]] else True
        #     for path in paths]

        assert (len(all_paths_single) == len(paths_to_objs_single))
        # assert (len(all_paths_single) == len(path_classes_single))

        return all_paths_single, paths_to_objs_single, []












def cut_off(all_paths,
                paths_to_objs,
                path_classes, label,
                paths, gt, anisotropy,ratio_true=0.13,ratio_false=0.4):

    """only selects and filters the paths where the ratio of absolute
    length to the length of the path outside the main label is below
    ratio_true and above ratio_false and classifies them as true or false """

    # print "cutting off..."
    # collects the underlying ground truth label for every point of every path
    gt_paths=[[gt[point[0],point[1],point[2]] for point in path]
              for path in paths]


    # collects the "length" for every point of every path
    len_paths=[np.concatenate(([norm3d(path[0],path[1])/2],
                               [norm3d(path[idx-1],path[idx])/2 +
                                norm3d(path[idx],path[idx+1])/2
                                for idx,point in enumerate(path)
                               if idx!=0 and idx!=len(path)-1],
                               [norm3d(path[-2],path[-1])/2]))
               for path in paths]



    gt_max_paths=[np.unique(val,return_counts=True) for val in gt_paths]

    #sort out the true ones which contain only one label
    indexes_true1=[idx for idx,val in enumerate(gt_max_paths) if len(val[0])==1]

    indexes_unknown = [idx for idx, val in enumerate(gt_max_paths) if len(val[0]) != 1]



    sums_unknown = [[idx, sum(len_paths[idx][gt_paths[idx]==
                                gt_max_paths[idx][0][np.argmax(gt_max_paths[idx][1])]]),
                        sum(len_paths[idx])] for idx in indexes_unknown]


    indexes_false=[idx for (idx,main_length,whole_length) in sums_unknown if
                   (whole_length-main_length)/whole_length>ratio_false]
    indexes_true2 = [idx for (idx, main_length, whole_length) in sums_unknown if
                     (whole_length - main_length) / whole_length < ratio_true]

    # concatenate the first true indices and the second
    indexes_true=np.concatenate((indexes_true1,indexes_true2)).tolist()


    [all_paths.append(np.array(paths[int(idx)])) for idx in indexes_true]
    [paths_to_objs.append(label) for x in xrange(0,len(indexes_true))]
    [path_classes.append(True) for x in xrange(0, len(indexes_true))]

    [all_paths.append(np.array(paths[int(idx)])) for idx in indexes_false]
    [paths_to_objs.append(label) for x in xrange(0, len(indexes_false))]
    [path_classes.append(False) for x in xrange(0, len(indexes_false))]

    return all_paths, paths_to_objs, path_classes
