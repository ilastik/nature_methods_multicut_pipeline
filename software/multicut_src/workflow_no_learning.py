import vigra
import numpy as np

from MCSolverImpl import weight_z_edges, weight_all_edges, weight_xyz_edges
from MCSolver import run_mc_solver
from DataSet import DataSet
from ExperimentSettings import ExperimentSettings
from tools import cacher_hdf5

# if build from sorce and not a conda pkg, we assume that we have cplex
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty # conda version build with cplex
    except ImportError:
        try:
            import nifty_wit_gurobi as nifty # conda version build with gurobi
        except ImportError:
            raise ImportError("No valid nifty version was found.")

# get weights from accumulated affinities between the local superpixel edges
# feature determines how to accumulated (supported: max, mean, median, quantile75, quantile90)
# TODO with defect correction
#@cacher_hdf5()
def multicut_costs_from_affinities_no_learning(ds,
        seg_id,
        inp_ids,
        feature = 'max',
        beta    = .5,
        weighting_scheme = 'z',
        weight = 16, # FIXME this is a magic parameter determining the strength of the weighting, I once determined 16 as a good value by grid search (on a different dataset....), this should be determined again!
        with_defcts = False
        ):
    assert seg_id < ds.n_seg
    assert len(inp_ids) == 2
    assert inp_ids[0] < ds.n_inp
    assert inp_ids[1] < ds.n_inp
    print "Computing mc costs from affinities"

    # map the feature we use to the index returned by rag.accumulateEdgeStatistics
    feat_to_index = {'mean' : 0, 'max' : 3, 'median' : 9, 'quantile75' : 10, 'quantile90' : 11}
    assert feature in feat_to_index.keys()

    index = feat_to_index[feature]

    rag = ds._rag(seg_id)
    aff_xy = ds.inp(inp_ids[0])
    aff_z  = ds.inp(inp_ids[1])

    edge_indications = ds.edge_indications(seg_id)

    print "Accumulating xy affinities with feature:", feature
    indicator_xy = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, aff_xy)
    costs      = rag.accumulateEdgeStatistics(indicator_xy)[:,index]

    print "Accumulating z affinities with feature:", feature
    # TODO change acccumulation to only accumulate the relevant pixels for z edges
    indicator_z = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, aff_z)
    costs_z   = rag.accumulateEdgeStatistics(indicator_z)[:,index]
    assert costs.shape[0] == costs_z.shape[0], "%i, %i" % (costs.shape[0], costs_z.shape[0])
    assert costs.shape[0] == edge_indications.shape[0], "%s, %s" % (str(costs.shape), str(edge_indications.shape) )

    # split xy and z edges accordingly (0 indicates z-edges !)
    costs[edge_indications == 0] = costs_z[edge_indications == 0]

    # make sure that we are in a valid range of values
    assert (costs >= 0.).all(), str(costs.min())
    assert (costs <= 1.).all(), str(costs.max())

    print "Cost range before scaling:", costs.min(), costs.max()

    # map affinities to weight space

    # first, scale to 0.001, 1. - 0.001 to avoid diverging costs
    c_min = 0.001
    c_max = 1. - c_min
    costs = (c_max - c_min) * costs + c_min

    # then map to ]-inf, inf[
    # NOTE I think the probabilities are inverted here compared to the normal case, that's why we need to invert the first log
    #costs = np.log( (1. - costs) / costs ) + np.log( (1. - beta) / beta )
    costs = np.log( costs / (1. - costs) ) + np.log( (1. - beta) / beta )
    assert not np.isinf(costs).any()
    print "Cost range after scaling:", costs.min(), costs.max()

    edge_sizes = rag.edgeLengths()

    # weight with the edge lens according to the weighting scheme
    if weighting_scheme == "z":
        print "Weighting Z edges"
        costs = weight_z_edges(costs, edge_sizes, edge_indications, weight)
    elif weighting_scheme == "xyz":
        print "Weighting xyz edges"
        costs = weight_xyz_edges(costs, edge_sizes, edge_indications, weight)
    elif weighting_scheme == "all":
        print "Weighting all edges"
        costs = weight_all_edges(costs, edge_sizes, weight)
    else:
        print "Edges are not weighted"

    assert not np.isinf(costs).any()
    return costs


# calculate the costs for lifted edges from the costs of the local edges
# via agglomerating costs along the shortest local paths
def lifted_multicut_costs_from_shortest_paths(
        ds,
        seg_id,
        local_costs,
        lifted_uv_ids,
        agglomeration_method = 'max'
        ):
    agglomerators = {
        'max' : np.max,
        'min' : np.min,
        'mean' : np.mean,
        'sum' : np.sum
    }
    agglomerator = agglomerators[agglomeration_method]
    assert agglomeration_method in agglomerators.keys() # TODO more ?!

    uv_ids = ds._adjacent_segments(seg_id)
    graph = nifty.graph.UndirectedGraph(uv_ids.max() + 1)

    lifted_costs = np.zeros(lifted_uv_ids.shape[0], dtype = local_costs.dtype)
    # process the lifted uv-ids s.t. we can run singleSourceMultiTarget
    # TODO parallelize this on the c++ end
    shortest_path = nifty.graph.ShortestPathDijkstra(graph)
    for u in np.unique(lifted_uv_ids[:,0]):
        where_u = np.where(lifted_uv_ids[:,0] == u)
        targets = lifted_uv_ids[where_u][:,1].tolist()
        paths = shortest_path.runSingleSourceMultiTarget(local_costs, u, targets)
        lifted_costs[where_u] = np.array( [agglomerator(pp) for pp in paths], dtype = local_costs.dtype )
    return lifted_costs


# TODO implement with defects
# multicut on the test dataset, weights learned with a rf on the train dataset
def multicut_workflow_no_learning(
        ds_test,
        seg_id,
        inp_ids,
        feature,
        with_defects = False):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(ds_test, DataSet)

    print "Running Multicut with weights from affinities"

    # get all parameters for the multicut
    uv_ids = ds_test._adjacent_segments(seg_id)
    n_var  = uv_ids.max() + 1

    # weights for the multicut
    costs = multicut_costs_from_affinities_no_learning(
            ds_test,
            seg_id,
            inp_ids,
            feature,
            ExperimentSettings().beta_local,
            ExperimentSettings().weighting_scheme,
            ExperimentSettings().weight,
            with_defects
        )

    return run_mc_solver(n_var, uv_ids, costs, mc_params)


def mala_clustering_workflow(
        ds,
        seg_id,
        inp_ids,
        threshold,
        use_edge_lens = False):
    assert len(inp_ids) == 2

    rag = ds._rag(seg_id)
    edge_indications = ds.edge_indications(seg_id)
    edge_lens        = ds.topology_features(seg_id, False)[:,0]

    # get the max affinities for xy edges from xy affinities
    aff_xy = ds.inp(inp_ids[0])
    edge_map_xy  = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, aff_xy)
    indicators = rag.accumulateEdgeStatistics(edge_map_xy)[:,3] # 3 -> max

    # get the max affinities for z edges from z affinities
    # TODO change acccumulation to only accumulate the relevant pixels for z edges
    aff_z  = ds.inp(inp_ids[1])
    edge_map_z  = vigra.graphs.implicitMeanEdgeMap(rag.baseGraph, aff_z)
    indicators_z = rag.accumulateEdgeStatistics(edge_map_z)[:,3] # 3 -> max

    # merge the indicators for xy and z edges
    indicators[edge_indications==1] = indicators_z[edge_indications==1]

    # run mala clustering
    graph = nifty.graph.UndirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())

    policy = nifty.graph.agglo.malaClusterPolicy(
            graph = graph,
            edgeIndicators = indicators,
            edgeSizes = edge_lens if use_edge_len else np.ones(graph.numberOfEdges),
            nodeSizes = np.ones(graph.numberOfNodes), # TODO we could also set this
            threshold = threshold)

    clustering = nifty.graph.agglo.agglomerativeClustering(policy)
    clustering.run()

    # get the node results and relabel it consecutively
    nodes = clustering.result()
    nodes, _, _ = vigra.analyss.relabelConsecutive(nodes, start_label = 0, keep_zeros = False)
    return nodes
