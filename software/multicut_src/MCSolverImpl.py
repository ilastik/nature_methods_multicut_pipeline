import numpy as np
import vigra
import opengm
import os
import time
from DataSet import DataSet, InverseCutout
from EdgeRF import *
from Tools import cacher_hdf5
import sys


###
### Functions for edgeprobabilities to edge energies
###

# calculate the energies for the multicut from membrane probabilities
@cacher_hdf5(ignoreNumpyArrays=True)
def probs_to_energies(ds, edge_probs, seg_id, exp_params):

    # scale the probabilities
    # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
    p_min = 0.001
    p_max = 1. - p_min
    edge_probs = (p_max - p_min) * edge_probs + p_min

    # probabilities to energies, second term is boundary bias
    edge_energies = np.log( (1. - edge_probs) / edge_probs ) + np.log( (1. - exp_params.beta_local) / exp_params.beta_local )

    # weight edges
    if exp_params.weighting_scheme == "z":
        print "Weighting Z edges"
        edge_energies = weight_z_edges(ds, edge_energies, seg_id, exp_params.weight)
    elif exp_params.weighting_scheme == "xyz":
        print "Weighting xyz edges"
        edge_energies = weight_xyz_edges(ds, edge_energies, seg_id, exp_params.weight)
    elif exp_params.weighting_scheme == "all":
        print "Weighting all edges"
        edge_energies = weight_all_edges(ds, edge_energies, seg_id, exp_params.weight)

    return edge_energies



def lifted_probs_to_energies(ds, edge_probs, edgeZdistance,
        betaGlobal=0.5, gamma=1.): # TODO weight connections in plane  kappa=20):

    p_min = 0.001
    p_max = 1. - p_min
    edge_probs = (p_max - p_min) * edge_probs + p_min

    # probabilities to energies, second term is boundary bias
    e = np.log( (1. - edge_probs) / edge_probs ) + np.log( (1. - betaGlobal) / betaGlobal )

    # additional weighting
    e /= gamma

    # weight down the z - edges with increasing distance
    if edgeZdistance is not None:
        e /= (edgeZdistance + 1.)

    return e


# weight z edges with their area
def weight_z_edges(ds, edge_energies, seg_id, weight):
    edge_areas       = ds._rag(seg_id).edgeLengths()
    edge_indications = ds.edge_indications(seg_id)
    assert edge_areas.shape[0] == edge_energies.shape[0]
    assert edge_indications.shape[0] == edge_energies.shape[0]

    energies_return = np.zeros_like(edge_energies)
    # z - edges are indicated with 0 !
    area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )

    # we only weight the z edges !
    w = weight * edge_areas[edge_indications == 0] / area_z_max
    energies_return[edge_indications == 0] = np.multiply(w, edge_energies[edge_indications == 0])
    energies_return[edge_indications == 1] = edge_energies[edge_indications == 1]

    return energies_return


# weight z edges with their area and xy edges with their length
# this is (probably) better than treating xy and z edges the same
def weight_xyz_edges(ds, edge_energies, seg_id, weight):
    edge_areas       = ds._rag(seg_id).edgeLengths()
    edge_indications = ds.edge_indications(seg_id)
    assert edge_areas.shape[0] == edge_energies.shape[0]
    assert edge_indications.shape[0] == edge_energies.shape[0]

    energies_return = np.zeros_like(edge_energies)

    # z - edges are indicated with 0 !
    area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
    len_xy_max = float( np.max( edge_areas[edge_indications == 1] ) )

    # weight the z edges !
    w_z = weight * edge_areas[edge_indications == 0] / area_z_max
    energies_return[edge_indications == 0] = np.multiply(w_z, edge_energies[edge_indications == 0])

    w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
    energies_return[edge_indications == 1] = np.multiply(w_xy, edge_energies[edge_indications == 1])

    return energies_return


# weight all edges with their length / area irrespective of them being xy or z
# note that this is the only weighting we can do for 3d-superpixel !
def weight_all_edges(ds, edge_energies, seg_id, weight):
    edge_areas       = ds._rag(seg_id).edgeLengths()
    assert edge_areas.shape[0] == edge_energies.shape[0]

    energies_return = np.zeros_like(edge_energies)

    area_max = float( np.max( edge_areas ) )
    w = weight * edge_areas / area_max
    energies_return = np.multiply(w, edge_energies)

    return energies_return


# solve the multicut problem with the exact opengm solver
def multicut_exact(n_var, uv_ids,
        edge_energies, exp_params):

    assert uv_ids.shape[0] == edge_energies.shape[0], str(uv_ids.shape[0]) + " , " + str(edge_energies.shape[0])
    assert np.max(uv_ids) == n_var - 1, str(np.max(uv_ids)) + " , " + str(n_var - 1)

    # set up the opengm model
    states = np.ones(n_var) * n_var
    gm = opengm.gm(states)

    # potts model
    potts_shape = [n_var, n_var]

    potts = opengm.pottsFunctions(potts_shape,
                                  np.zeros_like( edge_energies ),
                                  edge_energies )

    # potts model to opengm function
    fids_b = gm.addFunctions(potts)

    gm.addFactors(fids_b, uv_ids)

    # save the opengm model
    if False:
        opengm.saveGm(gm, "./gm_small_sample_C_gt.gm")

    # the workflow, we use
    wf = "(IC)(CC-IFD)"

    param = opengm.InfParam( workflow = wf, verbose = exp_params.verbose,
            verboseCPLEX = exp_params.verbose, numThreads = 4 )

    print "Starting Inference"

    inf = opengm.inference.Multicut(gm, parameter=param)
    t_inf = time.time()
    inf.infer()
    t_inf = time.time() - t_inf

    res_node = inf.arg()
    ru = res_node[uv_ids[:,0]]
    rv = res_node[uv_ids[:,1]]
    res_edge = ru!=rv

    E_glob = gm.evaluate(res_node)

    return (res_node, res_edge, E_glob, t_inf)


# solve the multicut problem with the nifty fusion moves solver
def multicut_fusionmoves(n_var, uv_ids,
        edge_energies, exp_params):

    assert uv_ids.shape[0] == edge_energies.shape[0], str(uv_ids.shape[0]) + " , " + str(edge_energies.shape[0])
    assert np.max(uv_ids) == n_var - 1, str(np.max(uv_ids)) + " , " + str(n_var - 1)

    # set up the opengm model
    states = np.ones(n_var) * n_var
    gm = opengm.gm(states)

    # potts model
    potts_shape = [n_var, n_var]

    potts = opengm.pottsFunctions(potts_shape,
                                  np.zeros_like( edge_energies ),
                                  edge_energies )

    # potts model to opengm function
    fids_b = gm.addFunctions(potts)
    gm.addFactors(fids_b, uv_ids)

    pparam = opengm.InfParam(seedFraction= exp_params.seed_fraction)
    parameter = opengm.InfParam(generator='randomizedWatershed',
                                proposalParam=pparam,
                                numStopIt=exp_params.num_it_stop,
                                numIt=exp_params.num_it)

    print "Starting Inference"
    inf = opengm.inference.IntersectionBased(gm, parameter=parameter)

    if exp_params.verbose:
        t_inf = time.time()
        inf.infer(inf.verboseVisitor())
        t_inf = time.time() - t_inf
    else:
        t_inf = time.time()
        inf.infer()
        t_inf = time.time() - t_inf

    res_node = inf.arg()
    ru = res_node[uv_ids[:,0]]
    rv = res_node[uv_ids[:,1]]
    res_edge = ru!=rv

    E_glob = gm.evaluate(res_node)

    return res_node, res_edge, E_glob, t_inf


def nifty_exact(n_var, uv_ids, edge_energies, exp_params):

    import nifty

    assert uv_ids.shape[0] == edge_energies.shape[0], str(uv_ids.shape[0]) + " , " + str(edge_energies.shape[0])
    assert np.max(uv_ids) == n_var - 1, str(np.max(uv_ids)) + " , " + str(n_var - 1)

    g =  nifty.graph.UndirectedGraph(int(n_var))
    g.insertEdges(uv_ids)

    assert g.numberOfEdges == edge_energies.shape[0]
    assert g.numberOfEdges == uv_ids.shape[0]

    obj = nifty.graph.multicut.multicutObjective(g, edge_energies)

    t_inf = time.time()

    solver = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
    ).create(obj)

    ret = solver.optimize()

    t_inf = time.time() - t_inf

    mc_energy = obj.evalNodeLabels(ret)

    return ret, mc_energy, t_inf



def nifty_fusionmoves(n_var, uv_ids, edge_energies, exp_params,nThreads=0,returnObj=False):

    import nifty

    assert uv_ids.shape[0] == edge_energies.shape[0], str(uv_ids.shape[0]) + " , " + str(edge_energies.shape[0])
    assert np.max(uv_ids) == n_var - 1, str(np.max(uv_ids)) + " , " + str(n_var - 1)

    g =  nifty.graph.UndirectedGraph(int(n_var))
    g.insertEdges(uv_ids)

    assert g.numberOfEdges == edge_energies.shape[0]
    assert g.numberOfEdges == uv_ids.shape[0]

    obj = nifty.graph.multicut.multicutObjective(g, edge_energies)

    greedy = obj.greedyAdditiveFactory().create(obj)
    ret    = greedy.optimize()

    t_inf = time.time()

    ilpFac = obj.multicutIlpFactory(ilpSolver='cplex',verbose=0,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
    )

    factory = obj.fusionMoveBasedFactory(
        verbose=1,
        fusionMove=obj.fusionMoveSettings(mcFactory=ilpFac),
        proposalGen=obj.watershedProposals(sigma=10,seedFraction=exp_params.seed_fraction),
        numberOfIterations=exp_params.num_it,
        numberOfParallelProposals=2*nThreads,
        numberOfThreads=nThreads,
        stopIfNoImprovement=exp_params.num_it_stop,
        fuseN=2,
    )

    solver = factory.create(obj)

    if exp_params.verbose:
        visitor = obj.multicutVerboseVisitor(1)
        ret = solver.optimize(nodeLabels=ret)
    else:
        ret = solver.optimize(nodeLabels=ret)

    t_inf = time.time() - t_inf

    mc_energy = obj.evalNodeLabels(ret)
    if not returnObj:
        return ret, mc_energy, t_inf
    else:
        return ret, mc_energy, t_inf, obj
