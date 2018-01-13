from itertools import combinations
import numpy as np
import vigra
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

from .DataSet import DataSet
from .EdgeRF import learn_and_predict_rf_from_gt, RandomForest
from .ExperimentSettings import ExperimentSettings
from .MCSolver import _get_feat_str
from .MCSolverImpl import probs_to_energies
from .lifted_mc import lifted_probs_to_energies, lifted_hard_gt, mask_lifted_edges, optimize_lifted
from .lifted_mc import learn_and_predict_lifted_rf
from .sparse_lifted_features import sparse_lifted_edges_and_features
from .tools.numpy_tools import find_matching_row_indices


def learn_sparse_lifted_rf(trainsets, seg_id_train):
    if isinstance(trainsets, DataSet):
        trainsets = [trainsets]

    features = []
    labels = []
    for ds_train in trainsets:
        uvs, feats = sparse_lifted_edges_and_features(ds_train, seg_id_train)
        labs = lifted_hard_gt(ds_train, seg_id_train, uvs)
        mask = mask_lifted_edges(ds_train, seg_id_train, labs, uvs, with_defects=False)
        features.append(feats[mask])
        labels.append(labs[mask])

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    assert features.shape[0] == labels.shape[0]
    rf = RandomForest(features, labels,
                      n_trees=ExperimentSettings().n_trees,
                      n_threads=ExperimentSettings().n_threads)
    return rf


def learn_and_predict_sparse_lifted_rf(trainsets, ds_test,
                                       seg_id_train, seg_id_test):
    # learn rf from sparse features
    rf = learn_sparse_lifted_rf(trainsets, seg_id_train)

    # get test uv-ids and features
    uvs_test, feats_test = sparse_lifted_edges_and_features(ds_test, seg_id_test)
    # predict with rf
    p_test = rf.predict_probabilities(feats_test)[:, 1]
    return p_test, uvs_test


def sparse_lifted_workflow(trainsets, ds_test,
                           seg_id_train, seg_id_test,
                           local_feature_list,
                           lifted_feature_list=['mito'],
                           w_sparse=1., w_lifted=1.):
    assert isinstance(ds_test, DataSet)
    assert isinstance(trainsets, (DataSet, list, tuple))
    print("Running sparse lifted multicut on", ds_test.ds_name)

    # probabilities and costs for local edges
    p_local = learn_and_predict_rf_from_gt(trainsets, ds_test,
                                           seg_id_train, seg_id_test,
                                           local_feature_list,
                                           with_defects=False,
                                           use_2rfs=ExperimentSettings().use_2rfs)
    edge_costs_local = probs_to_energies(ds_test, p_local, seg_id_test,
                                         ExperimentSettings().weighting_scheme,
                                         ExperimentSettings().weight,
                                         ExperimentSettings().beta_local,
                                         _get_feat_str(local_feature_list))

    # probabilities and costs for sparse lifted edges
    p_test_sparse, uv_ids_sparse = learn_and_predict_sparse_lifted_rf(trainsets, ds_test,
                                                                      seg_id_train, seg_id_test)
    # TODO make weight_z_distance settable
    edge_z_distance = None
    edge_costs_sparse = lifted_probs_to_energies(ds_test, p_test_sparse,
                                                 seg_id_test, edge_z_distance,
                                                 ExperimentSettings().lifted_neighborhood,
                                                 ExperimentSettings().beta_lifted,
                                                 gamma=w_sparse)

    # probabilities and costs for default lifted edges (if given)
    if lifted_feature_list:
        p_test_lifted, uv_ids_lifted = learn_and_predict_lifted_rf(trainsets,
                                                                   ds_test,
                                                                   seg_id_train,
                                                                   seg_id_test,
                                                                   feature_list_local=local_feature_list,
                                                                   feature_list_lifted=lifted_feature_list)
        edge_costs_lifted = lifted_probs_to_energies(ds_test, p_test_lifted,
                                                     seg_id_test, edge_z_distance,
                                                     ExperimentSettings().lifted_neighborhood,
                                                     ExperimentSettings().beta_lifted,
                                                     gamma=w_lifted)
        edges_total = len(edge_costs_local) + len(edge_costs_sparse) + len(edge_costs_lifted)
        # normalize all edges
        edge_costs_local *= (edges_total / len(edge_costs_local))
        edge_costs_sparse *= (edges_total / len(edge_costs_sparse))
        edge_costs_lifted *= (edges_total / len(edge_costs_lifted))

        # merge sparse and lifted edges:
        # for duplicates, keep sparse edges
        duplicates = find_matching_row_indices(uv_ids_lifted, uv_ids_sparse)[:, 0]
        if duplicates.size:
            # invert the mask to get the unique lifted edges
            unique_mask = np.ones(len(uv_ids_lifted), dtype='bool')
            unique_mask[duplicates] = False
            uv_ids_lifted = uv_ids_lifted[duplicates]
            edge_costs_lifted = edge_costs_lifted[duplicates]
        uv_ids_sparse = np.concatenate([uv_ids_sparse, uv_ids_lifted], axis=0)
        edge_costs_sparse = np.concatenate([edge_costs_lifted, edge_costs_sparse], axis=0)
    else:
        edges_total = len(edge_costs_local) + len(edge_costs_sparse)
        # normalize all edges
        edge_costs_local *= (edges_total / len(edge_costs_local))
        edge_costs_sparse *= (edges_total / len(edge_costs_sparse))

    uvs_local = ds_test.uv_ids(seg_id_test)
    node_labels, e_lifted, t_lifted = optimize_lifted(uvs_local,
                                                      uv_ids_sparse,
                                                      edge_costs_local,
                                                      edge_costs_sparse,
                                                      starting_point=None)
    edge_labels = node_labels[uvs_local[:, 0]] != node_labels[uvs_local[:, 1]]
    return node_labels, edge_labels, e_lifted, t_lifted


def resolve_object(nodes, node_prior_list, multicut_costs, rag):
    # find the nodes with priors in this object
    nodes_priors1 = np.in1d(nodes, node_prior_list[0])
    nodes_priors2 = np.in1d(nodes, node_prior_list[1])
    assert nodes_priors1.size and nodes_priors2.size

    # find the multicut weights corresponding to this object
    edge_ids = rag.edgesFromNodeList(nodes.tolist())
    local_costs = multicut_costs[edge_ids]
    local_uvs = rag.uvIds()[edge_ids]

    # map to consecutive node labeling
    local_nodes, mapping, _ = vigra.analysis.relabelConsecutive(nodes)
    # map local_uvs, local_weights and node priors
    # TODO
    nodes_priors1 = [mapping[npr] for npr in nodes_priors1]
    nodes_priors2 = [mapping[npr] for npr in nodes_priors2]

    # introduce repulsive lifted edges between priors of different types
    lifted_uvs = np.array(combinations(nodes_priors1, nodes_priors2), dtype='uint32')
    # TODO how dow we determine the strenght of the lifted repulstion
    repl_strenght = 50.
    lifted_costs = repl_strenght * np.ones(len(lifted_uvs))

    # solve the new lifted model
    return optimize_lifted(local_uvs, lifted_uvs,
                           local_costs, lifted_costs,
                           starting_point=None)[0]


def sparse_lifted_topdown(ds,
                          seg_id,
                          result,
                          multicut_costs,
                          prior_threshold=.5):
    seg = ds.seg(seg_id)
    # vesicle and dendrite pixelwise maps
    pix_maps = [ds.inp(2), ds.inp(3)]
    # find the mapping of over-segmentation nodes to the result segmentation
    rag = ds.rag(seg_id)
    node_mapping = np.array(nrag.gridRagAccumulateLabels(rag, result), dtype='uint32')
    object_ids = np.unique(result)
    n_objects = len(object_ids)

    # get the inverse mapping of objects in the result to oversegmentation nodes
    # TODO this could be done more efficiently
    object_mapping = {obj_id: node_mapping[node_mapping == obj_id]
                      for obj_id in object_ids}

    # iterate over the pixel-wise maps and find oversegmentation nodes
    # with priors and the corresponding mapping of priors to objects
    # find objects that have overlap with the priors
    node_prior_list = []
    objects_with_prior = np.zeros((n_objects, 2), dtype='bool')
    for i, pf in enumerate(pix_maps):
        acc_feats = vigra.analysis.extractRegionFeatures(pf, seg, ["maximum"])
        acc_max = acc_feats["maximum"]
        nodes_with_prior = acc_max > prior_threshold
        node_prior_list.append(nodes_with_prior)
        objects_with_prior[node_mapping[nodes_with_prior], i] = True

    # find objects with contradicting priors (i.e. objects that
    # contain nodes with both prior types)
    contradicting_priors = objects_with_prior.all(axis=1)
    objects_to_resolve = object_ids[contradicting_priors]
    resolved_nodes = node_mapping.copy()
    offset = object_ids.max() + 1

    print("Found %i / %i objects that have contradicting priors and will be resolved"
          % (len(objects_to_resolve), n_objects))

    # TODO parallelize with threadpool !
    # resolve the objects that were found
    for obj_id in objects_to_resolve:
        nodes = object_mapping[obj_id]
        resolved = resolve_object(nodes, node_prior_list, multicut_costs, rag)
        n_new_objs = resolved.max() + 1
        resolved += offset
        resolved_nodes[nodes] = resolved
        offset += n_new_objs

    vigra.analysis.relabelConsecutive(resolved_nodes, out=resolved_nodes)
    return resolved_nodes
