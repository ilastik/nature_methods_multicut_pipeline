import numpy as np
import vigra
import os
import time
from DataSet import DataSet, InverseCutout
from Tools import cacher_hdf5
import sys

# if build from sorce and not a conda pkg, we assume that we have cplex
try:
    import nifty
    ilp_bkend = 'cplex'
except ImportError:
    try:
        import nifty_with_cplex as nifty # conda version build with cplex
        ilp_bkend = 'cplex'
    except ImportError:
        try:
            import nifty_wit_gurobi as nifty # conda version build with gurobi
            ilp_bkend = 'gurobi'
        except ImportError:
            raise ImportError("No valid nifty version was found.")



###
### Functions for edgeprobabilities to edge energies
###

# calculate the energies for the multicut from membrane probabilities
# the last argument is only for caching correctly with different feature combinations
@cacher_hdf5(ignoreNumpyArrays=True)
def probs_to_energies(ds, edge_probs, seg_id, exp_params, feat_cache):

    # scale the probabilities
    # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
    p_min = 0.001
    p_max = 1. - p_min
    edge_probs = (p_max - p_min) * edge_probs + p_min

    # probabilities to energies, second term is boundary bias
    edge_energies = np.log( (1. - edge_probs) / edge_probs ) + np.log( (1. - exp_params.beta_local) / exp_params.beta_local )

    if exp_params.weighting_scheme in ("z", "xyz", "all"):
        edge_areas       = ds._rag(seg_id).edgeLengths()
        edge_indications = ds.edge_indications(seg_id)

    # weight edges
    if exp_params.weighting_scheme == "z":
        print "Weighting Z edges"
        edge_energies = weight_z_edges(ds, edge_energies, seg_id, edge_areas, edge_indications, exp_params.weight)
    elif exp_params.weighting_scheme == "xyz":
        print "Weighting xyz edges"
        edge_energies = weight_xyz_edges(ds, edge_energies, seg_id, edge_areas, edge_indications, exp_params.weight)
    elif exp_params.weighting_scheme == "all":
        print "Weighting all edges"
        edge_energies = weight_all_edges(ds, edge_energies, seg_id, edge_areas, exp_params.weight)

    # set the edges with the segmask to be maximally repulsive
    if ds.has_seg_mask:
        uv_ids = ds._adjacent_segments(seg_id)
        ignore_mask = (uv_ids == 0).any(axis = 1)
        edge_energies[ ignore_mask ] = 2 * edge_energies.min()

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
def weight_z_edges(ds, edge_energies, seg_id, edge_areas, edge_indications, weight):
    assert edge_areas.shape[0] == edge_energies.shape[0], "%s, %s" % (str(edge_areas.shape), str(edge_energies.shape))
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
def weight_xyz_edges(ds, edge_energies, seg_id, edge_areas, edge_indications, weight):
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
def weight_all_edges(ds, edge_energies, seg_id, edge_areas, weight):
    assert edge_areas.shape[0] == edge_energies.shape[0]

    energies_return = np.zeros_like(edge_energies)

    area_max = float( np.max( edge_areas ) )
    w = weight * edge_areas / area_max
    energies_return = np.multiply(w, edge_energies)

    return energies_return


def multicut_exact(n_var,
        uv_ids,
        edge_energies,
        exp_params):

    assert uv_ids.shape[0] == edge_energies.shape[0], str(uv_ids.shape[0]) + " , " + str(edge_energies.shape[0])
    assert np.max(uv_ids) == n_var - 1, str(np.max(uv_ids)) + " , " + str(n_var - 1)

    g = nifty.graph.UndirectedGraph(int(n_var))
    g.insertEdges(uv_ids)

    assert g.numberOfEdges == edge_energies.shape[0]
    assert g.numberOfEdges == uv_ids.shape[0]

    obj = nifty.graph.multicut.multicutObjective(g, edge_energies)

    t_inf = time.time()

    solver = obj.multicutIlpFactory(ilpSolver=ilp_bkend,verbose=0,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
    ).create(obj)

    if exp_params.verbose:
        visitor = obj.multicutVerboseVisitor(1)
        ret = solver.optimize(visitor=visitor)
    else:
        ret = solver.optimize()

    t_inf = time.time() - t_inf

    mc_energy = obj.evalNodeLabels(ret)

    return ret, mc_energy, t_inf, obj



def multicut_fusionmoves(n_var,
        uv_ids,
        edge_energies,
        exp_params,
        nThreads=0,
        returnObj=False):

    assert uv_ids.shape[0] == edge_energies.shape[0], str(uv_ids.shape[0]) + " , " + str(edge_energies.shape[0])
    assert np.max(uv_ids) == n_var - 1, str(np.max(uv_ids)) + " , " + str(n_var - 1)

    g = nifty.graph.UndirectedGraph(int(n_var))
    g.insertEdges(uv_ids)

    assert g.numberOfEdges == edge_energies.shape[0]
    assert g.numberOfEdges == uv_ids.shape[0]

    obj = nifty.graph.multicut.multicutObjective(g, edge_energies)

    greedy = obj.greedyAdditiveFactory().create(obj)
    ret    = greedy.optimize()

    t_inf = time.time()

    ilpFac = obj.multicutIlpFactory(ilpSolver=ilp_bkend,verbose=0,
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
        ret = solver.optimize(nodeLabels=ret,visitor=visitor)
    else:
        ret = solver.optimize(nodeLabels=ret)

    t_inf = time.time() - t_inf

    mc_energy = obj.evalNodeLabels(ret)
    if not returnObj:
        return ret, mc_energy, t_inf
    else:
        return ret, mc_energy, t_inf, obj
