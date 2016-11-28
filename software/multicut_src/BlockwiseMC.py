import numpy as np
import opengm
import time

from DataSet import DataSet
from EdgeRF import learn_and_predict_rf_from_gt
from MCSolverImpl import probs_to_energies, multicut_exact, multicut_fusionmoves
from ExperimentSettings import ExperimentSettings
from Tools import UnionFind

from concurrent import futures

def blockwise_multicut_workflow(ds_train, ds_test,
        seg_id_train, seg_id_test,
        offset_list, feature_list,
        mc_params):
    assert isinstance(ds_train, DataSet) or isinstance(ds_train, list)
    assert isinstance(ds_test, DataSet)
    assert isinstance(mc_params, ExperimentSettings)

    # get the energies for the whole test block
    edge_probs = learn_and_predict_rf_from_gt(mc_params.rf_cache_folder,
            ds_train, ds_test,
            seg_id_train, seg_id_test,
            feature_list, mc_params)

    edge_energies = probs_to_energies(ds_test, edge_probs,
            seg_id_test, mc_params)

    seg_global = ds_test.seg(seg_id_test)
    rag_global = ds_test._rag(seg_id_test)

    # number of nodes and edges in the orginial problem
    n_edges = rag_global.edgeNum
    n_nodes = rag_global.nodeNum
    uv_ids_glob = ds_test._adjacent_segments(seg_id_test)

    assert edge_energies.shape[0] == n_edges, str(edge_energies.shape[0]) + " , " + str(n_edges)

    # set up the global graphical model to get the energy later
    # set up the opengm model
    states = np.ones(n_nodes) * n_nodes
    gm_global = opengm.gm(states)
    # potts model
    potts_shape = [n_nodes, n_nodes]
    potts = opengm.pottsFunctions(potts_shape,
                                  np.zeros_like( edge_energies ),
                                  edge_energies )
    # potts model to opengm function
    fids_b = gm_global.addFunctions(potts)
    gm_global.addFactors(fids_b, uv_ids_glob)

    # save the opengm model
    if False:
        opengm.saveGm(gm_global, "./gm_large_" + ds_test.ds_name + ".gm")

    # implementation of a single blockwise mc
    def block_mc(edge_energies, seg_local, rag_global, mc_params):

        # get the nodes in this blocks
        nodes = np.unique(seg_local)
        # global nodes to local nodes
        global_to_local_nodes = {}
        for i in xrange(nodes.shape[0]):
            global_to_local_nodes[nodes[i]] = i

        # get the edges and uvids in this block
        inner_edges  = []
        outer_edges  = []
        uv_ids_local = {}
        for n_id in nodes:
            node = rag_global.nodeFromId( long(n_id) )
            for adj_node in rag_global.neighbourNodeIter(node):
                edge = rag_global.findEdge(node,adj_node)
                if adj_node.id in nodes:
                    inner_edges.append(edge.id)
                    u_local = global_to_local_nodes[n_id]
                    v_local = global_to_local_nodes[adj_node.id]
                    uv_ids_local[edge.id] = [ u_local, v_local ]
                else:
                    outer_edges.append(edge.id)

        # need to get rid of potential duplicates and order uv-ids propperly
        inner_edges = np.unique(inner_edges)
        outer_edges = np.unique(outer_edges)
        uv_ids      = np.zeros( (inner_edges.shape[0], 2), dtype = np.uint32 )
        for i in xrange( inner_edges.shape[0] ):
            edge_id   = inner_edges[i]
            uv_ids[i] = uv_ids_local[edge_id]
        uv_ids = np.sort(uv_ids, axis = 1)

        assert uv_ids.max() == nodes.shape[0] - 1, str(uv_ids.max()) + " , " +str(nodes.shape[0] - 1)

        n_var = nodes.shape[0]
        energies_local = edge_energies[inner_edges]

        node_result, edge_result, _, _ = multicut_fusionmoves(n_var, uv_ids,
                energies_local, mc_params)

        return edge_result, inner_edges, outer_edges


    # solve all blocks in paralllel
    t_inf = time.time()
    nWorkers = min( len(offset_list), mc_params.n_threads )
    #nWorkers = 4
    with futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
        tasks = []
        for p in offset_list:
            assert seg_global.shape[0] >= p[1] and seg_global.shape[1] >= p[3] and seg_global.shape[2] >= p[5]
            tasks.append( executor.submit( block_mc, edge_energies,
                seg_global[p[0]:p[1],p[2]:p[3],p[4]:p[5]],
                rag_global, mc_params) )

    blockwise_results = [future.result() for future in tasks]
    assert len(blockwise_results) == len(offset_list)

    cut_edges = np.zeros( n_edges, dtype = np.uint32 )

    for edge_result in blockwise_results:

        cut_edges[edge_result[2]] += 1
        cut_edges[edge_result[1]] += edge_result[0]

    # all edges which are cut at least once will be cut
    cut_edges[cut_edges >= 1] = 1

    # merge nodes according to cut edges
    # this means, that we merge all segments that have an edge with value 0 in between
    # for this, we use a ufd datastructure
    udf = UnionFind( n_nodes )

    assert uv_ids_glob.shape[0] == cut_edges.shape[0]
    merge_nodes = uv_ids_glob[cut_edges == 0]

    for merge_pair in merge_nodes:
        u = merge_pair[0]
        v = merge_pair[1]
        udf.merge(u, v)

    # we need to get the result of the merging
    new_to_old_nodes = udf.get_merge_result()
    # number of nodes for the new problem
    n_nodes_new = len(new_to_old_nodes)

    # find old to new nodes
    old_to_new_nodes = np.zeros( n_nodes, dtype = np.uint32 )
    for set_id in xrange( n_nodes_new ):
        for n_id in new_to_old_nodes[set_id]:
            assert n_id < n_nodes, str(n_id) + " , " + str(n_nodes)
            old_to_new_nodes[n_id] = set_id

    # find new edges and new edge weights
    active_edges = np.where( cut_edges == 1 )[0]
    new_edges_dict = {}
    for edge_id in active_edges:
        u_old = uv_ids_glob[edge_id][0]
        v_old = uv_ids_glob[edge_id][1]
        n_0_new = old_to_new_nodes[u_old]
        n_1_new = old_to_new_nodes[v_old]
        # we have to bew in different new nodes!
        assert n_0_new != n_1_new, str(n_0_new) + " , " + str(n_1_new)
        # need to order to always have the same keys
        u_new = min(n_0_new, n_1_new)
        v_new = max(n_0_new, n_1_new)
        # need to check if have already come by this new edge
        if (u_new,v_new) in new_edges_dict:
            new_edges_dict[(u_new,v_new)] += edge_energies[edge_id]
        else:
            new_edges_dict[(u_new,v_new)] = edge_energies[edge_id]

    n_edges_new = len( new_edges_dict.keys() )
    uv_ids_new = np.array( new_edges_dict.keys() )
    assert uv_ids_new.shape[0] == n_edges_new, str(uv_ids_new.shape[0]) + " , " + str(n_edges_new)
    # this should have the correct order
    energies_new = np.array( new_edges_dict.values() )

    print "Merging of blockwise results reduced problemsize:"
    print "Nodes: From", n_nodes, "to", n_nodes_new
    print "Edges: From", n_edges, "to", n_edges_new
    print "Running MC for reduced problem"

    # run mc on the new problem
    if mc_params.solver == "opengm_exact":
        res_node_new, res_edge_new, E_new, _ = multicut_exact(
            n_nodes_new, uv_ids_new,
            energies_new, mc_params)
    elif mc_params.solver == "opengm_fusionmoves":
        res_node_new, res_edge_new, E_new, _ = multicut_fusionmoves(
            n_nodes_new, uv_ids_new,
            energies_new, mc_params)

    assert res_node_new.shape[0] == n_nodes_new, str(res_node_new.shape[0]) + " , " + str(n_nodes_new)

    # project back to old problem
    res_node = np.zeros(n_nodes, dtype = np.uint32)
    for n_id in xrange(res_node_new.shape[0]):
        for old_node in new_to_old_nodes[n_id]:
            res_node[old_node] = res_node_new[n_id]

    t_inf = time.time() - t_inf

    ru = res_node[uv_ids_glob[:,0]]
    rv = res_node[uv_ids_glob[:,1]]
    res_edge = ru!=rv

    # get the global energy
    E_glob = gm_global.evaluate(res_node)

    if 0 in res_node:
        res_node += 1

    return res_node, res_edge, E_glob, t_inf
