import numpy as np
import vigra
import os
import time
import sys

from DataSet import DataSet
from ExperimentSettings import ExperimentSettings

from MCSolverImpl import probs_to_energies, multicut_exact, multicut_fusionmoves
from EdgeRF import learn_and_predict_rf_from_gt
from lifted_mc import learn_and_predict_lifted_rf, optimizeLifted
from defect_handling import modified_adjacency, modified_probs_to_energies

import graph as agraph

def _get_feat_str(feature_list):
    feat_str = ""
    if "raw" in feature_list:
        feat_str += "raw"
    if "prob" in feature_list:
        feat_str += "prob"
    if "affinity" in feature_list:
        feat_str += "affinity"
    if "reg" in feature_list:
        feat_str += "reg"
    if "topo" in feature_list:
        feat_str += "topo"
    return feat_str

def run_mc_solver(n_var, uv_ids, edge_energies):

    # solve the multicut witht the given solver
    if ExperimentSettings().solver == "multicut_exact":
        mc_node, mc_energy, t_inf = multicut_exact(n_var, uv_ids, edge_energies)
    elif ExperimentSettings().solver == "multicut_fusionmoves":
        mc_node, mc_energy, t_inf = multicut_fusionmoves(n_var, uv_ids, edge_energies, ExperimentSettings().n_threads)
    else:
        raise RuntimeError("Something went wrong, sovler " + ExperimentSettings().solver + ", not in valid solver.")

    # get the result mapped to the edges
    ru = mc_node[uv_ids[:,0]]
    rv = mc_node[uv_ids[:,1]]
    mc_edges = ru!=rv

    # we dont want zero as a segmentation result
    # because it is the ignore label in many settings
    mc_node, _, _ = vigra.analysis.relabelConsecutive(mc_node, start_label = 1, keep_zeros = False)
    assert len(mc_node) == n_var, "%i, %i" % (len(mc_node), n_var)
    return mc_node, mc_edges, mc_energy, t_inf


# multicut on the test dataset, weights learned with a rf on the train dataset
def multicut_workflow(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)
    assert isinstance(ds_test, DataSet)

    print "Running multicut on", ds_test.ds_name
    if isinstance(trainsets, DataSet):
        print "Weights learned on", trainsets.ds_name
    else:
        print "Weights learned on multiple Datasets"
    print "with solver", ExperimentSettings().solver

    # get edge probabilities from random forest
    print "Learning random forests with", ExperimentSettings().n_trees, "trees"
    edge_probs = learn_and_predict_rf_from_gt(
            trainsets,
            ds_test,
            seg_id_train,
            seg_id_test,
            feature_list,
            with_defects = False,
            use_2rfs = ExperimentSettings().use_2rfs)

    # get all parameters for the multicut
    # number of variables = number of nodes
    seg_id_max = ds_test.seg(seg_id_test).max()
    n_var = seg_id_max + 1
    # uv - ids = node ides connected by the edges
    uv_ids = ds_test._adjacent_segments(seg_id_test)
    assert n_var == uv_ids.max() + 1, "%i, %i" % (n_var, uv_ids.max() + 1)
    # energies for the multicut
    edge_energies = probs_to_energies(ds_test,
            edge_probs,
            seg_id_test,
            ExperimentSettings().weighting_scheme,
            ExperimentSettings().weight,
            ExperimentSettings().beta_local,
            _get_feat_str(feature_list))
    return run_mc_solver(n_var, uv_ids, edge_energies)


# multicut on the test dataset, weights learned with a rf on the train dataset
def multicut_workflow_with_defect_correction(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)
    assert isinstance(ds_test, DataSet)

    # make sure that the test dataset has defects
    # (need to call mod adjacency to make sure the flag is set correctly)
    modified_adjacency(ds_test, seg_id_test)
    assert ds_test.has_defects

    print "Running multicut with defect correction on", ds_test.ds_name
    if isinstance(trainsets, DataSet):
        print "Weights learned on", trainsets.ds_name
    else:
        print "Weights learned on multiple Datasets"
    print "with solver", ExperimentSettings().solver

    # get edge probabilities from random forest
    edge_probs = learn_and_predict_rf_from_gt(
            trainsets,
            ds_test,
            seg_id_train,
            seg_id_test,
            feature_list,
            with_defects = True,
            use_2rfs = ExperimentSettings().use_2rfs)

    # get all parameters for the multicut
    uv_ids = modified_adjacency(ds_test, seg_id_test)
    n_var = uv_ids.max() + 1

    # energies for the multicut
    edge_energies = modified_probs_to_energies(
            ds_test,
            edge_probs,
            seg_id_test,
            uv_ids,
            ExperimentSettings().weighting_scheme,
            ExperimentSettings().weight,
            ExperimentSettings().beta_local,
            _get_feat_str(feature_list) )
    return run_mc_solver(n_var, uv_ids, edge_energies)


# lifted multicut on the test dataset, weights learned with a rf on the train dataset
def lifted_multicut_workflow(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list_local,
        feature_list_lifted,
        gamma = 1.,
        warmstart = False,
        weight_z_lifted = True):

    assert isinstance(ds_test, DataSet)
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)

    print "Running lifted multicut on", ds_test.ds_name
    if isinstance(trainsets, DataSet):
        print "Weights learned on", trainsets.ds_name
    else:
        print "Weights learned on multiple datasets"

    #) step one, train a random forest
    print "Start learning"

    pTestLifted, uv_ids_lifted, nzTest = learn_and_predict_lifted_rf(
            trainsets,
            ds_test,
            seg_id_train,
            seg_id_test,
            feature_list_lifted,
            feature_list_local)

    # get edge probabilities from random forest on the complete training set
    pTestLocal = learn_and_predict_rf_from_gt(
            trainsets,
            ds_test,
            seg_id_train,
            seg_id_test,
            feature_list_local,
            with_defects = False,
            use_2rfs = ExperimentSettings().use_2rfs)

    feat_str = _get_feat_str(feature_list_local)
    # energies for the multicut
    edge_energies_local = probs_to_energies(
            ds_test,
            pTestLocal,
            seg_id_test,
            ExperimentSettings().weighting_scheme,
            ExperimentSettings().weight,
            ExperimentSettings().beta_local,
            _get_feat_str(feature_list_local))

    # node z to edge z distance
    edgeZdistance = np.abs( nzTest[uv_ids_lifted[:,0]] - nzTest[uv_ids_lifted[:,1]] ) if weight_z_lifted else None
    edge_energies_lifted = lifted_probs_to_energies(
            ds_test,
            pTestLifted,
            seg_id_test,
            edgeZdistance,
            ExperimentSettings().lifted_neighborhood,
            ExperimentSettings().beta_global,
            gamma)

    # weighting edges with their length for proper lifted to local scaling
    edge_energies_local  /= edge_energies_local.shape[0]
    edge_energies_lifted /= edge_energies_lifted.shape[0]

    print "build lifted model"
    # remove me in functions
    uvs_local = ds_test._adjacent_segments(seg_id_test)

    # warmstart with multicut result
    if warmstart:
        n_var_mc = uvs_local.max() + 1
        mc_nodes, mc_edges, _, _ = run_mc_solver(n_var_mc, uv_ids_local, edge_energies_local)
        starting_point = mc_nodes[uv_ids_lifted[:,0]] != mc_nodes[uv_ids_lifted[:,1]]
    else:
        starting_point = None

    print "optimize"
    nodeLabels = optimizeLifted(uvs_local, uv_ids_lifted,
            edge_energies_local, edge_energies_lifted,
            starting_point)

    edgeLabels = nodeLabels[uvs_local[:,0]] != nodeLabels[uvs_local[:,1]]
    return nodeLabels, edgeLabels, -14, 100


# lifted multicut on the test dataset, weights learned with a rf on the train dataset
def lifted_multicut_workflow_with_defect_correction(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list_local,
        feature_list_lifted,
        gamma = 1.,
        warmstart = False,
        weight_z_lifted = True):

    assert isinstance(ds_test, DataSet)
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)

    # make sure that the test dataset has defects
    # (need to call mod adjacency to make sure the flag is set correctly)
    modified_adjacency(ds_test, seg_id_test)
    assert ds_test.has_defects

    print "Running lifted multicut with defect detection on", ds_test.ds_name
    if isinstance(trainsets, DataSet):
        print "Weights learned on", trainsets.ds_name
    else:
        print "Weights learned on multiple datasets"

    #) step one, train a random forest
    print "Start learning"

    pTestLifted, uv_ids_lifted, nzTest = learn_and_predict_lifted_rf(
            trainsets,
            ds_test,
            seg_id_train,
            seg_id_test,
            feature_list_lifted,
            feature_list_local,
            with_defects = True)

    # get edge probabilities from random forest on the complete training set
    pTestLocal = learn_and_predict_rf_from_gt(
            trainsets,
            ds_test,
            seg_id_train,
            seg_id_test,
            feature_list_local,
            with_defcts = True,
            use_2rfs = ExperimentSettings().use_2rfs)

    # get all parameters for the multicut
    uv_ids_local = modified_adjacency(ds_test, seg_id_test)
    n_var = uv_ids_local.max() + 1

    # energies for the multicut
    edge_energies_local = modified_probs_to_energies(
            ds_test,
            pTestLocal,
            seg_id_test,
            uv_ids_local,
            ExperimentSettings().weighting_scheme,
            ExperimentSettings().weight,
            ExperimentSettings().beta_local,
            _get_feat_str(feature_list_local) )
    assert not np.isnan(edge_energies_local).any()

    # lifted energies
    # node z to edge z distance
    edgeZdistance = np.abs( nzTest[uv_ids_lifted[:,0]] - nzTest[uv_ids_lifted[:,1]] ) if weight_z_lifted else None
    edge_energies_lifted = lifted_probs_to_energies(
            ds_test,
            pTestLifted,
            seg_id_test,
            edgeZdistance,
            ExperimentSettings().lifted_neighborhood,
            gamma = gamma,
            betaGlobal = ExperimentSettings().beta_global,
            with_defects = True)
    assert not np.isnan(edge_energies_lifted).any()

    # weighting edges with their length for proper lifted to local scaling
    edge_energies_local  /= edge_energies_local.shape[0]
    edge_energies_lifted /= edge_energies_lifted.shape[0]

    print "build lifted model"
    # remove me in functions
    originalGraph = agraph.Graph(uv_ids_local.max() + 1)
    originalGraph.insertEdges(uv_ids_local)
    model = agraph.liftedMcModel(originalGraph)

    # set cost for local edges
    model.setCosts(uv_ids_local,edge_energies_local)
    # set cost for lifted edges
    model.setCosts(uv_ids_lifted, edge_energies_lifted)

    # warmstart with multicut result
    if warmstart:
        mc_nodes, mc_edges, _, _ = run_mc_solver(n_var_mc, uv_ids_local, edge_energies_local)
        uvTotal = model.liftedGraph().uvIds()
        starting_point = mc_nodes[uvTotal[:,0]] != mc_nodes[uvTotal[:,1]]
    else:
        starting_point = None

    print "optimize"
    nodeLabels = optimizeLifted(uv_ids_local, uv_ids_lifted,
            edge_energies_local, edge_energies_lifted,
            starting_point)
    edgeLabels = nodeLabels[uv_ids_local[:,0]]!=nodeLabels[uv_ids_local[:,1]]
    return nodeLabels, edgeLabels, -14, 100
