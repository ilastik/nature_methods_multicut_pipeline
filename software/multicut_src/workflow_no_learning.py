import vigra
import numpy as np

from MCSolverImpl import weight_z_edges, weight_all_edges, weight_xyz_edges
from MCSolver import run_mc_solver
from DataSet import DataSet
from ExperimentSettings import ExperimentSettings
from lifted_mc import optimize_lifted, compute_and_save_lifted_nh

# for future uses
# from tools import cacher_hdf5

# if build from sorce and not a conda pkg, we assume that we have cplex
try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex as nifty  # conda version build with cplex
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi as nifty  # conda version build with gurobi
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


def accumulate_affinities_over_edges(
        ds,
        seg_id,
        inp_ids,
        feature,
        z_direction=2,
        rag=None
):

    assert seg_id < ds.n_seg
    assert len(inp_ids) == 2
    assert inp_ids[0] < ds.n_inp
    assert inp_ids[1] < ds.n_inp

    # map the feature we use to the index returned by rag.accumulateEdgeStatistics
    feat_to_index = {'mean': 0, 'max': 8, 'median': 5, 'quantile75': 6, 'quantile90': 7}
    assert feature in feat_to_index.keys()

    index = feat_to_index[feature]

    if rag is None:
        rag = ds.rag(seg_id)

    aff_xy = ds.inp(inp_ids[0])
    aff_z  = ds.inp(inp_ids[1])

    edge_indications = ds.edge_indications(seg_id)

    print "Accumulating xy affinities with feature:", feature
    accumulated = nrag.accumulateEdgeFeaturesFlat(
        rag,
        aff_xy,
        aff_xy.min(),
        aff_xy.max(),
        z_direction,
        ExperimentSettings().n_threads
    )[:, index]

    print "Accumulating z affinities with feature:", feature
    accumulated_z = nrag.accumulateEdgeFeaturesFlat(
        rag,
        aff_z,
        aff_z.min(),
        aff_z.max(),
        z_direction,
        ExperimentSettings().n_threads
    )[:, index]
    assert accumulated.shape == accumulated_z.shape, "%s, %s" % (accumulated.shape, accumulated_z.shape)
    assert accumulated.shape[0] == edge_indications.shape[0], \
        "%s, %s" % (str(accumulated.shape), str(edge_indications.shape))

    # split xy and z edges accordingly (0 indicates z-edges !)
    accumulated[edge_indications == 0] = accumulated_z[edge_indications == 0]

    return accumulated


# get weights from accumulated affinities between the local superpixel edges
# feature determines how to accumulated (supported: max, mean, median, quantile75, quantile90)
# TODO with defect correction
# @cacher_hdf5()
def costs_from_affinities(
        ds,
        seg_id,
        inp_ids,
        feature='max',
        beta=.5,
        weighting_scheme='z',
        # FIXME this is a magic parameter determining the strength of the weighting,
        # I once determined 16 as a good value by grid search (on a different dataset....),
        # this should be determined again!
        weight=16,
        with_defcts=False,
        return_probs=False
):

    print "Computing mc costs from affinities"
    # NOTE we need to invert, because we have affinity maps, not boundary probabilities
    probs = 1. - accumulate_affinities_over_edges(
        ds,
        seg_id,
        inp_ids,
        feature,
        ExperimentSettings().affinity_z_direction
    )

    # make sure that we are in a valid range of values
    assert (probs >= 0.).all(), str(probs.min())
    assert (probs <= 1.).all(), str(probs.max())

    print "Cost range before scaling:", probs.min(), probs.max()
    costs = probs.copy()

    # map affinities to weight space
    # first, scale to 0.001, 1. - 0.001 to avoid diverging costs
    c_min = 0.001
    c_max = 1. - c_min
    costs = (c_max - c_min) * costs + c_min

    # then map to ]-inf, inf[
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
    assert not np.isinf(costs).any()
    print "Cost range after scaling:", costs.min(), costs.max()

    edge_sizes = ds.topology_features(seg_id, False)[:, 0]
    edge_indications = ds.edge_indications(seg_id)

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

    # if we have a seg mask set edges to the ignore segment to be max repulsive
    if ds.has_seg_mask:
        uv_ids = ds.uv_ids(seg_id)
        max_repulsive = 2 * costs.min()
        ignore_ids = (uv_ids == ExperimentSettings().ignore_seg_value).any(axis=1)
        costs[ignore_ids] = max_repulsive

    if return_probs:
        return costs, probs
    else:
        return costs


# TODO cache...
# calculate the costs for lifted edges from the costs of the local edges
# via agglomerating costs along the shortest local paths
def lifted_multicut_costs_from_shortest_paths(
        ds,
        seg_id,
        local_probabilities,
        lifted_uv_ids,
        agglomeration_method='max',
        gamma=1,
        beta_lifted=0.5,
        edge_z_distance=None
):

    # TODO add quantiles ?!
    agglomerators = {
        'max': np.max,
        'min': np.min,
        'mean': np.mean
    }
    agglomerator = agglomerators[agglomeration_method]
    assert agglomeration_method in agglomerators.keys()

    uv_ids = ds.uv_ids(seg_id)
    graph = nifty.graph.UndirectedGraph(uv_ids.max() + 1)

    lifted_costs = np.zeros(lifted_uv_ids.shape[0], dtype=local_probabilities.dtype)

    # process the lifted uv-ids s.t. we can run singleSourceMultiTarget
    # TODO use parallel shortest path version
    shortest_path = nifty.graph.ShortestPathDijkstra(graph)
    for u in np.unique(lifted_uv_ids[:, 0]):
        where_u = np.where(lifted_uv_ids[:, 0] == u)
        targets = lifted_uv_ids[where_u][:, 1].tolist()
        paths = shortest_path.runSingleSourceMultiTarget(local_probabilities, u, targets, returnNodes=False)
        assert len(paths) == len(targets)

        # agglomerate based on local_probabilities !
        # TODO there is nifty functionality for this but maybe it is a good idea
        # to expose this in the shortest path !
        lifted_costs[where_u] = np.array(
            [agglomerator(local_probabilities[pp]) for pp in paths],
            dtype=local_probabilities.dtype
        )
        print u
        print
        print targets
        print
        print paths
        print
        print lifted_costs[where_u]
        quit()

    # TODO probabilities to costs
    p_min = 0.001
    p_max = 1. - p_min
    lifted_costs = (p_max - p_min) * lifted_costs + p_min

    # probabilities to energies, second term is boundary bias
    lifted_costs = np.log((1. - lifted_costs) / lifted_costs) + np.log((1. - beta_lifted) / beta_lifted)

    # additional weighting
    lifted_costs /= gamma

    # weight down the z - edges with increasing distance
    if edge_z_distance is not None:
        assert edge_z_distance.shape[0] == lifted_costs.shape[0], \
            "%s, %s" % (str(edge_z_distance.shape), str(lifted_costs.shape))
        lifted_costs /= (edge_z_distance + 1.)

    # set the edges within the segmask to be maximally repulsive
    # these should all be removed, check !
    if ds.has_seg_mask:
        assert np.sum((lifted_uv_ids == ExperimentSettings().ignore_seg_value).any(axis=1)) == 0

    return lifted_costs


# TODO implement with defects
# multicut on the test dataset, weights learned with a rf on the train dataset
def multicut_workflow_no_learning(
        ds_test,
        seg_id,
        inp_ids,
        feature,
        with_defects=False
):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(ds_test, DataSet)

    print "Running Multicut with weights from affinities"

    # get all parameters for the multicut
    uv_ids = ds_test.uv_ids(seg_id)
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
        with_defects
    )

    return run_mc_solver(n_var, uv_ids, costs)


# TODO implement with defects
# multicut on the test dataset, weights learned with a rf on the train dataset
def lifted_multicut_workflow_no_learning(
        ds_test,
        seg_id,
        inp_ids,
        local_feature,
        lifted_feature,
        gamma=1,
        with_defects=False
):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(ds_test, DataSet)

    print "Running Lifted Multicut with weights from affinities"

    # get all parameters for the multicut
    uv_ids = ds_test.uv_ids(seg_id)

    # weights for the multicut
    local_costs, local_probs = costs_from_affinities(
        ds_test,
        seg_id,
        inp_ids,
        local_feature,
        ExperimentSettings().beta_local,
        ExperimentSettings().weighting_scheme,
        ExperimentSettings().weight,
        with_defects,
        return_probs=True
    )

    lifted_uv_ids = compute_and_save_lifted_nh(
        ds_test,
        seg_id,
        ExperimentSettings().lifted_neighborhood,
        with_defects
    )

    # TODO weight z lifted edges with their distance
    lifted_costs = lifted_multicut_costs_from_shortest_paths(
        ds_test,
        seg_id,
        local_probs,
        lifted_uv_ids,
        lifted_feature,
        gamma,
        ExperimentSettings().beta_lifted
    )

    return optimize_lifted(uv_ids, lifted_uv_ids, local_costs, lifted_costs)


def mala_clustering_workflow(
        ds,
        seg_id,
        inp_ids,
        threshold,
        use_edge_lens=False
):
    assert len(inp_ids) == 2
    import nifty.graph.agglo as nagglo

    # need to invert due to affinities!
    indicators = 1. - accumulate_affinities_over_edges(
        ds,
        seg_id,
        inp_ids,
        'max',
        ExperimentSettings().affinity_z_direction
    )
    edge_lens  = ds.topology_features(seg_id, False)[:, 0]

    # run mala clustering
    uv_ids = ds.uv_ids(seg_id)
    graph = nifty.graph.UndirectedGraph(uv_ids.max() + 1)
    graph.insertEdges(uv_ids)

    if ds.has_seg_mask:
        ignore_ids = (uv_ids != ExperimentSettings().ignore_seg_value).all(axis=1)
        indicators[ignore_ids] = 1.

    # policy = nagglo.edgeWeightedClusterPolicy(
    policy = nagglo.malaClusterPolicy(
        graph=graph,
        edgeIndicators=indicators,
        edgeSizes=edge_lens.astype('float32') if use_edge_lens else np.ones(graph.numberOfEdges, dtype='float32'),
        nodeSizes=np.ones(graph.numberOfNodes, dtype='float32'),  # TODO we could also set this
        threshold=threshold
    )

    print "start clustering"
    clustering = nifty.graph.agglo.agglomerativeClustering(policy)
    clustering.run(verbose=True)

    # get the node results and relabel it consecutively
    nodes = clustering.result()
    nodes, _, _ = vigra.analysis.relabelConsecutive(nodes, start_label=0, keep_zeros=False)
    return nodes
