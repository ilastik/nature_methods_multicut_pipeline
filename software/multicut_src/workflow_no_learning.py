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

def accumulate_affinities_over_edges(
        ds,
        seg_id,
        inp_ids,
        feature,
        z_direction = 2,
        rag = None
        ):

    assert seg_id < ds.n_seg
    assert len(inp_ids) == 2
    assert inp_ids[0] < ds.n_inp
    assert inp_ids[1] < ds.n_inp

    # map the feature we use to the index returned by rag.accumulateEdgeStatistics
    feat_to_index = {'mean' : 0, 'max' : 8, 'median' : 5, 'quantile75' : 6, 'quantile90' : 7}
    assert feature in feat_to_index.keys()

    index = feat_to_index[feature]

    if rag is None:
        rag = ds.nifty_rag(seg_id)
    to_nifty_order = ds.vigra_to_nifty(seg_id)
    to_vigra_order = ds.nifty_to_vigra(seg_id)

    # TODO we only need the ascontiguous when still supporting vigra
    aff_xy = np.ascontiguousarray(ds.inp(inp_ids[0]).transpose((2,1,0)))
    aff_z  = np.ascontiguousarray(ds.inp(inp_ids[1]).transpose((2,1,0)))

    edge_indications = ds.edge_indications(seg_id)[to_nifty_order]

    print "Accumulating xy affinities with feature:", feature
    accumulated = nifty.graph.rag.accumulateEdgeFeaturesFlat(
            rag,
            aff_xy,
            aff_xy.min(),
            aff_xy.max(),
            z_direction,
            ExperimentSettings().n_threads)[:,index]

    print "Accumulating z affinities with feature:", feature
    accumulated_z = nifty.graph.rag.accumulateEdgeFeaturesFlat(
            rag,
            aff_z,
            aff_z.min(),
            aff_z.max(),
            z_direction,
            ExperimentSettings().n_threads)[:,index]
    assert accumulated.shape[0] == accumulated_z.shape[0], "%i, %i" % (accumulated.shape[0], accumulated_z.shape[0])
    assert accumulated.shape[0] == edge_indications.shape[0], "%s, %s" % (str(accumulated.shape), str(edge_indications.shape) )

    # split xy and z edges accordingly (0 indicates z-edges !)
    accumulated[edge_indications == 0] = accumulated_z[edge_indications == 0]

    return accumulated[to_vigra_order]


# get weights from accumulated affinities between the local superpixel edges
# feature determines how to accumulated (supported: max, mean, median, quantile75, quantile90)
# TODO with defect correction
#@cacher_hdf5()
def costs_from_affinities(
        ds,
        seg_id,
        inp_ids,
        feature = 'max',
        beta    = .5,
        weighting_scheme = 'z',
        weight = 16, # FIXME this is a magic parameter determining the strength of the weighting, I once determined 16 as a good value by grid search (on a different dataset....), this should be determined again!
        with_defcts = False,
        z_direction = 2
        ):

    print "Computing mc costs from affinities"
    # NOTE we need to invert, because we have affinity maps, not boundary probabilities
    costs = 1. - accumulate_affinities_over_edges(ds, seg_id, inp_ids, feature, z_direction)

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
    costs = np.log( (1. - costs) / costs ) + np.log( (1. - beta) / beta )
    assert not np.isinf(costs).any()
    print "Cost range after scaling:", costs.min(), costs.max()

    return costs

    #edge_sizes = rag.edgeLengths()

    ## weight with the edge lens according to the weighting scheme
    #if weighting_scheme == "z":
    #    print "Weighting Z edges"
    #    costs = weight_z_edges(costs, edge_sizes, edge_indications, weight)
    #elif weighting_scheme == "xyz":
    #    print "Weighting xyz edges"
    #    costs = weight_xyz_edges(costs, edge_sizes, edge_indications, weight)
    #elif weighting_scheme == "all":
    #    print "Weighting all edges"
    #    costs = weight_all_edges(costs, edge_sizes, weight)
    #else:
    #    print "Edges are not weighted"

    #assert not np.isinf(costs).any()
    #return costs


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
    # TODO probabilities to costs
    return lifted_costs


# TODO implement with defects
# multicut on the test dataset, weights learned with a rf on the train dataset
def multicut_workflow_no_learning(
        ds_test,
        seg_id,
        inp_ids,
        feature,
        with_defects = False,
        z_direction  = 2):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(ds_test, DataSet)

    print "Running Multicut with weights from affinities"

    # get all parameters for the multicut
    uv_ids = ds_test._adjacent_segments(seg_id)
    n_var  = uv_ids.max() + 1

    # weights for the multicut
    costs = costs_from_affinities(
            ds_test,
            seg_id,
            inp_ids,
            feature,
            ExperimentSettings().beta_local,
            ExperimentSettings().weighting_scheme,
            ExperimentSettings().weight,
            with_defects,
            z_direction
        )

    # if we have a seg mask set edges to the ignore segment to be max repulsive
    if ds.has_seg_mask:
        max_repulsive = 2 * costs.min()
        ignore_ids = (uv_ids != ExperimentSettings().ignore_seg_value).all(axis=1)
        costs[ignore_ids] = max_repulsive

    return run_mc_solver(n_var, uv_ids, costs)


def mala_clustering_workflow(
        ds,
        seg_id,
        inp_ids,
        threshold,
        use_edge_lens = False,
        z_direction = 2 # this is the mala convention (z-edges encode for affinity from z+1 to z)
    ):
    assert len(inp_ids) == 2
    import nifty.graph.agglo as nagglo

    # need to invert due to affinities!
    indicators = 1. - accumulate_affinities_over_edges(ds, seg_id, inp_ids, 'max', z_direction)
    edge_lens  = ds.topology_features(seg_id, False)[:,0]

    # run mala clustering
    uv_ids = ds._adjacent_segments(seg_id)
    graph = nifty.graph.UndirectedGraph(uv_ids.max() + 1)
    graph.insertEdges(uv_ids)

    if ds.has_seg_mask:
        ignore_ids = (uv_ids != ExperimentSettings().ignore_seg_value).all(axis=1)
        indicators[ignore_ids] = 1.

    #policy = nagglo.edgeWeightedClusterPolicy(
    policy = nagglo.malaClusterPolicy(
            graph = graph,
            edgeIndicators = indicators,
            edgeSizes = edge_lens.astype('float32') if use_edge_lens else np.ones(graph.numberOfEdges, dtype = 'float32'),
            nodeSizes = np.ones(graph.numberOfNodes, dtype = 'float32'), # TODO we could also set this
            threshold = threshold
        )

    print "start clustering"
    clustering = nifty.graph.agglo.agglomerativeClustering(policy)
    clustering.run(verbose = True)

    # get the node results and relabel it consecutively
    nodes = clustering.result()
    nodes, _, _ = vigra.analysis.relabelConsecutive(nodes, start_label = 0, keep_zeros = False)
    return nodes
