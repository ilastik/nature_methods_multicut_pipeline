import numpy as np

from ExperimentSettings import ExperimentSettings
from MCSolverImpl import to_costs
from MCSolver import run_mc_solver
from tools import replace_from_dict, find_exclusive_matching_indices
from higher_order_feats import get_xy_junctions

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


def get_multicut_problem_2d(ds, seg_id, z, edge_probs):
    seg = ds.seg(seg_id)
    rag = ds.rag(seg_id)

    # extract the sub-graph in slice z
    # we could also use ds.node_z_coord
    nodes = np.unique(seg[z])
    inner_edges, outer_edges, sub_graph = rag.extractSubgraphFromNodes(nodes)

    # assert all inner_edges are xy-edges
    edge_indications = ds.edge_indications(seg_id)
    assert (edge_indications[inner_edges] == 1).all()

    # get the node mapping and uv-mapping
    node_mapping = {node: i for i, node in enumerate(nodes)}

    uv_ids = rag.uvIds()
    uv_z = replace_from_dict(uv_ids[inner_edges], node_mapping)
    assert uv_z.max() + 1 == len(nodes)

    probs_z = edge_probs[inner_edges]
    costs_z = to_costs(probs_z, ExperimentSettings().beta_local)

    # the weighting schemes don't make a difference for 2d multicut, so we apply the same weighting
    # if any weighting scheme was chosen
    weighting_scheme = ExperimentSettings().weighting_scheme
    if weighting_scheme in ('xyz', 'z', 'all'):

        weight = ExperimentSettings().weight
        edge_lens = nrag.accumulateEdgeMeanAndLength(
            rag,
            np.zeros(seg_z.shape, dtype='float32')  # fake data
        )[:, 1]
        w = weight * edge_lens / float(edge_lens.max())
        costs_z = np.multiply(w, costs_z)

    return uv_z, costs_z


def solve_multicut_2d(ds, seg_id, z, edge_probs):

    uv_z, costs_z = get_multicut_problem_2d(ds, seg_id, z, edge_probs)

    # solve the mc problem and return nodes
    n_var = uv_z.max() + 1
    mc_nodes, _, _, _ = run_mc_solver(n_var, uv_z, costs_z)
    return mc_nodes


def project_result_2d(ds, seg_id, z, node_result):
    seg = ds.seg(seg_id)
    rag = ds.rag(seg_id)
    seg_z = seg[z]

    # extract the sub-graph in slice z
    # we could also use ds.node_z_coord
    nodes = np.unique(seg_z)
    assert len(node_result) == len(nodes)

    node_mapping = {node: i for i, node in enumerate(nodes)}
    seg_z = replace_from_dict(seg_z, node_mapping)
    rag_z = nrag.gridRag(seg_z)

    return nrag.projectScalarNodeDataToPixels(rag_z, node_result)


def get_junction_ids_2d(ds, seg_id, z):
    seg = ds.seg(seg_id)
    rag = ds.rag(seg_id)

    junctions_to_edges = get_xy_junctions(ds, seg_id)

    # extract the sub-graph in slice z
    # we could also use ds.node_z_coord
    nodes = np.unique(seg[z])
    node_mapping = {node: i for i, node in enumerate(nodes)}

    inner_edges, outer_edges, sub_graph = rag.extractSubgraphFromNodes(nodes)

    # find the junction ids which have only edge-ids in the subgraph
    junction_ids = find_exclusive_matching_indices(junctions_to_edges, np.array(inner_edges))

    return junction_ids, inner_edges
