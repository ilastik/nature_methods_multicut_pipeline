import numpy as np
import vigra
import opengm
import os
import time
import sys

from DataSet import DataSet, InverseCutout
from ExperimentSettings import ExperimentSettings
from MCSolverImpl import *
from EdgeRF import *
from lifted_mc import *
from defect_handling import modified_mc_problem, modified_probs_to_energies

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

def run_mc_solver(n_var, uv_ids, edge_energies, mc_params):
    #vigra.writeHDF5(edge_energies, "./edge_energies_nproof_train.h5", "data")
    # solve the multicut witht the given solver
    if mc_params.solver == "opengm_exact":
        (mc_node, mc_edges, mc_energy, t_inf) = multicut_exact(
                n_var, uv_ids,
                edge_energies, mc_params)
    elif mc_params.solver == "opengm_fusionmoves":
        (mc_node, mc_edges, mc_energy, t_inf) = multicut_fusionmoves(
                n_var, uv_ids,
                edge_energies, mc_params)
    elif mc_params.solver == "nifty_exact":
        (mc_node, mc_energy, t_inf) = nifty_exact(
                n_var, uv_ids, edge_energies, mc_params)
        ru = mc_node[uv_ids[:,0]]
        rv = mc_node[uv_ids[:,1]]
        mc_edges = ru!=rv
    elif mc_params.solver == "nifty_fusionmoves":
        (mc_node, mc_energy, t_inf) = nifty_fusionmoves(
                n_var, uv_ids, edge_energies, mc_params)
        ru = mc_node[uv_ids[:,0]]
        rv = mc_node[uv_ids[:,1]]
        mc_edges = ru!=rv
    else:
        raise RuntimeError("Something went wrong, sovler " + mc_params.solver + ", not in valid solver.")
    # we dont want zero as a segmentation result
    # because it is the ignore label in many settings
    if 0 in mc_node:
        mc_node += 1
    return mc_node, mc_edges, mc_energy, t_inf


# multicut on the test dataset, weights learned with a rf on the train dataset
def multicut_workflow(ds_train, ds_test,
        seg_id_train, seg_id_test,
        feature_list, mc_params,
        use_2_rfs = False):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(ds_train, DataSet) or isinstance(ds_train, list)
    assert isinstance(ds_test, DataSet)
    assert isinstance(mc_params, ExperimentSettings )

    print "Running multicut on", ds_test.ds_name
    if isinstance(ds_train, DataSet):
        print "Weights learned on", ds_train.ds_name
    else:
        print "Weights learned on multiple Datasets"
    print "with solver", mc_params.solver

    # get edge probabilities from random forest
    if use_2_rfs:
        print "Learning separate random forests for xy - and z - edges with", mc_params.n_trees, "trees"
        edge_probs = learn_and_predict_anisotropic_rf(mc_params.rf_cache_folder,
                ds_train,
                ds_test,
                seg_id_train,
                seg_id_test,
                feature_list, feature_list,
                mc_params)
    else:
        print "Learning single random forest with", mc_params.n_trees, "trees"
        edge_probs = learn_and_predict_rf_from_gt(mc_params.rf_cache_folder,
                ds_train,
                ds_test,
                seg_id_train,
                seg_id_test,
                feature_list,
                mc_params)

    # for an InverseCutout, make sure that the artificial edges will be cut
    if isinstance(ds_test, InverseCutout):
        edge_probs[ds_test.get_artificial_edges(seg_id_test)] = 1.

    # get all parameters for the multicut
    # number of variables = number of nodes
    seg_id_max = ds_test.seg(seg_id_test).max()
    n_var = seg_id_max + 1
    assert n_var == ds_test._rag(seg_id_test).nodeNum
    # uv - ids = node ides connected by the edges
    uv_ids = ds_test._adjacent_segments(seg_id_test)
    # energies for the multicut
    edge_energies = probs_to_energies(ds_test,
            edge_probs,
            seg_id_test,
            mc_params,
            _get_feat_str(feature_list))
    return run_mc_solver(n_var, uv_ids, edge_energies, mc_params)


# multicut on the test dataset, weights learned with a rf on the train dataset
def multicut_workflow_with_defect_correction(ds_train, ds_test,
        seg_id_train, seg_id_test,
        feature_list, mc_params,
        n_bins, bin_threshold,
        use_2_rfs = False):
    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(ds_train, DataSet) or isinstance(ds_train, list)
    assert isinstance(ds_test, DataSet)
    assert isinstance(mc_params, ExperimentSettings )
    print "Running multicut with defect correction on", ds_test.ds_name
    if isinstance(ds_train, DataSet):
        print "Weights learned on", ds_train.ds_name
    else:
        print "Weights learned on multiple Datasets"
    print "with solver", mc_params.solver
    # get edge probabilities from random forest
    if use_2_rfs:
        print "Learning separate random forests for xy - and z - edges with", mc_params.n_trees, "trees"
        edge_probs = learn_and_predict_anisotropic_rf(mc_params.rf_cache_folder,
                ds_train,
                ds_test,
                seg_id_train,
                seg_id_test,
                feature_list, feature_list,
                mc_params, True,
                n_bins, bin_threshold)
    else:
        print "Learning single random forest with", mc_params.n_trees, "trees"
        edge_probs = learn_and_predict_rf_from_gt(mc_params.rf_cache_folder,
                ds_train,
                ds_test,
                seg_id_train,
                seg_id_test,
                feature_list,
                mc_params, True,
                n_bins, bin_threshold)
    # for an InverseCutout, make sure that the artificial edges will be cut
    if isinstance(ds_test, InverseCutout):
        raise AttributeError("Not supported for defect correction workflow.")
    # get all parameters for the multicut
    n_var, uv_ids = modified_mc_problem(ds_test, seg_id_test, n_bins, bin_threshold)
    # energies for the multicut
    edge_energies = modified_probs_to_energies(ds_test,
            edge_probs, seg_id_test, uv_ids,
            mc_params, n_bins, bin_threshold,
            _get_feat_str(feature_list))
    return run_mc_solver(n_var, uv_ids, edge_energies, mc_params)


# lifted multicut on the test dataset, weights learned with a rf on the train dataset
def lifted_multicut_workflow(ds_train, ds_test,
        seg_id_train, seg_id_test,
        feature_list_local, feature_list_lifted,
        mc_params, gamma = 1., warmstart = False, weight_z_lifted = True):
    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(ds_test, DataSet)
    assert isinstance(ds_train, DataSet) or isinstance(ds_train, list)
    assert isinstance(mc_params, ExperimentSettings )

    print "Running lifted multicut on", ds_test.ds_name
    if isinstance(ds_train, DataSet):
        print "Weights learned on", ds_train.ds_name
    else:
        print "Weights learned on multiple datasets"

    #) step one, train a random forest
    print "Start learning"

    pTestLifted, uvIds, nzTest = learn_and_predict_lifted(
            ds_train, ds_test,
            seg_id_train, seg_id_test,
            feature_list_lifted, feature_list_local,
            mc_params)

    # get edge probabilities from random forest on the complete training set
    pTestLocal = learn_and_predict_rf_from_gt(mc_params.rf_cache_folder,
        ds_train, ds_test,
        seg_id_train, seg_id_test,
        feature_list_local, mc_params)

    feat_str = _get_feat_str(feature_list_local)
    # energies for the multicut
    edge_energies_local = probs_to_energies(ds_test,
            pTestLocal, seg_id_test, mc_params, _get_feat_str(feature_list_local))

    # lifted energies
    if weight_z_lifted:
        # node z to edge z distance
        edgeZdistance = np.abs( nzTest[uvIds[:,0]] - nzTest[uvIds[:,1]] )
        edge_energies_lifted = lifted_probs_to_energies(ds_test,
            pTestLifted, edgeZdistance, gamma = gamma, betaGlobal = mc_params.beta_global)
    else:
        edge_energies_lifted = lifted_probs_to_energies(ds_test,
            pTestLifted, None, gamma = gamma, betaGlobal = mc_params.beta_global)

    # weighting edges with their length for proper lifted to local scaling
    edge_energies_local  /= edge_energies_local.shape[0]
    edge_energies_lifted /= edge_energies_lifted.shape[0]

    print "build lifted model"
    # remove me in functions
    rag = ds_test._rag(seg_id_test)
    originalGraph = agraph.Graph(rag.nodeNum)
    originalGraph.insertEdges(rag.uvIds())
    model = agraph.liftedMcModel(originalGraph)

    # set cost for local edges
    model.setCosts(rag.uvIds(),edge_energies_local)
    # set cost for lifted edges
    model.setCosts(uvIds, edge_energies_lifted)

    # warmstart with multicut result
    if warmstart:
        n_var_mc = ds_test.seg(seg_id_test).max() + 1
        mc_nodes, mc_edges = run_mc_solver(n_var_mc, ds_test._adjacent_segments(seg_id_test), edge_energies_local, mc_params)
        uvTotal = model.liftedGraph().uvIds()
        starting_point = mc_nodes[uvTotal[:,0]] != mc_nodes[uvTotal[:,1]]
    else:
        starting_point = None

    print "optimize"
    nodeLabels = optimizeLifted(ds_test, model, starting_point)
    edgeLabels = nodeLabels[rag.uvIds()[:,0]]!=nodeLabels[rag.uvIds()[:,1]]
    return nodeLabels, edgeLabels, -14, 100


# lifted multicut on the test dataset, weights learned with a rf on the train dataset
def lifted_multicut_workflow_with_defect_correction(ds_train, ds_test,
        seg_id_train, seg_id_test,
        feature_list_local, feature_list_lifted,
        mc_params, n_bins, bin_threshold,
        gamma = 1., warmstart = False, weight_z_lifted = True):

    assert isinstance(ds_test, DataSet)
    assert isinstance(ds_train, DataSet) or isinstance(ds_train, list)
    assert isinstance(mc_params, ExperimentSettings )

    print "Running lifted multicut with defect detection on", ds_test.ds_name
    if isinstance(ds_train, DataSet):
        print "Weights learned on", ds_train.ds_name
    else:
        print "Weights learned on multiple datasets"

    #) step one, train a random forest
    print "Start learning"

    pTestLifted, uvIds, nzTest = learn_and_predict_lifted(
            ds_train, ds_test,
            seg_id_train, seg_id_test,
            feature_list_lifted, feature_list_local,
            mc_params, True, n_bins, bin_threshold)

    # get edge probabilities from random forest on the complete training set
    pTestLocal = learn_and_predict_rf_from_gt(mc_params.rf_cache_folder,
        ds_train, ds_test,
        seg_id_train, seg_id_test,
        feature_list_local, mc_params,
        True, n_bins, bin_threshold)

    # energies for the multicut
    n_var_mc, uv_ids_local = modified_mc_problem(ds_test, seg_id_test, n_bins, bin_threshold)
    # energies for the multicut
    edge_energies_local = modified_probs_to_energies(ds_test,
            pTestLocal, seg_id_test, uv_ids_local,
            mc_params, n_bins, bin_threshold,
            _get_feat_str(feature_list_local))
    assert not np.isnan(edge_energies_local).any()

    # lifted energies
    if weight_z_lifted:
        # node z to edge z distance
        edgeZdistance = np.abs( nzTest[uvIds[:,0]] - nzTest[uvIds[:,1]] )
        edge_energies_lifted = lifted_probs_to_energies(ds_test,
            pTestLifted, edgeZdistance, gamma = gamma, betaGlobal = mc_params.beta_global)
    else:
        edge_energies_lifted = lifted_probs_to_energies(ds_test,
            pTestLifted, None, gamma = gamma, betaGlobal = mc_params.beta_global)
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
    model.setCosts(uvIds, edge_energies_lifted)

    # warmstart with multicut result
    if warmstart:
        mc_nodes, mc_edges = run_mc_solver(n_var_mc, uv_ids_local, edge_energies_local, mc_params)
        uvTotal = model.liftedGraph().uvIds()
        starting_point = mc_nodes[uvTotal[:,0]] != mc_nodes[uvTotal[:,1]]
    else:
        starting_point = None

    print "optimize"
    nodeLabels = optimizeLifted(ds_test, model, starting_point)
    edgeLabels = nodeLabels[uv_ids_local[:,0]]!=nodeLabels[uv_ids_local[:,1]]
    return nodeLabels, edgeLabels, -14, 100
