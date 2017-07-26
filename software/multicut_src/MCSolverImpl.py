import numpy as np
import time

from ExperimentSettings import ExperimentSettings
from tools import cacher_hdf5

# if build from sorce and not a conda pkg, we assume that we have cplex
try:
    import nifty
    ilp_bkend = 'cplex'
except ImportError:
    try:
        import nifty_with_cplex as nifty  # conda version build with cplex
        ilp_bkend = 'cplex'
    except ImportError:
        try:
            import nifty_with_gurobi as nifty  # conda version build with gurobi
            ilp_bkend = 'gurobi'
        except ImportError:
            raise ImportError("No valid nifty version was found.")


###
# Functions for edgeprobabilities to edge energies
###

# calculate the energies for the multicut from membrane probabilities
# the last argument is only for caching correctly with different feature combinations
@cacher_hdf5(ignoreNumpyArrays=True)
def probs_to_energies(
    ds,
    edge_probs,
    seg_id,
    weighting_scheme,
    weight,
    beta,
    feat_cache
):

    # scale the probabilities
    # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
    p_min = 0.001
    p_max = 1. - p_min
    edge_probs = (p_max - p_min) * edge_probs + p_min

    # probabilities to energies, second term is boundary bias
    edge_energies = np.log((1. - edge_probs) / edge_probs) + np.log((1. - beta) / beta)

    if weighting_scheme in ("z", "xyz", "all"):
        edge_areas       = ds.topology_features(seg_id, False)[:, 0].astype('uint32')
        edge_indications = ds.edge_indications(seg_id)

    # weight edges
    if weighting_scheme == "z":
        print "Weighting Z edges"
        edge_energies = weight_z_edges(edge_energies, edge_areas, edge_indications, weight)
    elif weighting_scheme == "xyz":
        print "Weighting xyz edges"
        edge_energies = weight_xyz_edges(edge_energies, edge_areas, edge_indications, weight)
    elif weighting_scheme == "all":
        print "Weighting all edges"
        edge_energies = weight_all_edges(edge_energies, edge_areas, weight)

    # set the edges within the segmask to be maximally repulsive
    if ds.has_seg_mask:
        uv_ids = ds.uv_ids(seg_id)
        ignore_mask = (uv_ids == ExperimentSettings().ignore_seg_value).any(axis=1)
        edge_energies[ignore_mask] = 2 * edge_energies.min()

    return edge_energies


# weight z edges with their area
def weight_z_edges(edge_energies, edge_areas, edge_indications, weight):
    assert edge_areas.shape[0] == edge_energies.shape[0], "%s, %s" % (str(edge_areas.shape), str(edge_energies.shape))
    assert edge_indications.shape[0] == edge_energies.shape[0]

    energies_return = np.zeros_like(edge_energies)
    # z - edges are indicated with 0 !
    area_z_max = float(np.max(edge_areas[edge_indications == 0]))

    # we only weight the z edges !
    w = weight * edge_areas[edge_indications == 0] / area_z_max
    energies_return[edge_indications == 0] = np.multiply(w, edge_energies[edge_indications == 0])
    energies_return[edge_indications == 1] = edge_energies[edge_indications == 1]

    return energies_return


# weight z edges with their area and xy edges with their length
# this is (probably) better than treating xy and z edges the same
def weight_xyz_edges(edge_energies, edge_areas, edge_indications, weight):
    assert edge_areas.shape[0] == edge_energies.shape[0]
    assert edge_indications.shape[0] == edge_energies.shape[0]

    energies_return = np.zeros_like(edge_energies)

    # z - edges are indicated with 0 !
    area_z_max = float(np.max(edge_areas[edge_indications == 0]))
    len_xy_max = float(np.max(edge_areas[edge_indications == 1]))

    # weight the z edges !
    w_z = weight * edge_areas[edge_indications == 0] / area_z_max
    energies_return[edge_indications == 0] = np.multiply(w_z, edge_energies[edge_indications == 0])

    w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
    energies_return[edge_indications == 1] = np.multiply(w_xy, edge_energies[edge_indications == 1])

    return energies_return


# weight all edges with their length / area irrespective of them being xy or z
# note that this is the only weighting we can do for 3d-superpixel !
def weight_all_edges(edge_energies, edge_areas, weight):
    assert edge_areas.shape[0] == edge_energies.shape[0]

    energies_return = np.zeros_like(edge_energies)

    area_max = float(np.max(edge_areas))
    w = weight * edge_areas / area_max
    energies_return = np.multiply(w, edge_energies)

    return energies_return


def nifty_objective(n_var, uv_ids, edge_energies):
    assert uv_ids.shape[0] == edge_energies.shape[0], str(uv_ids.shape[0]) + " , " + str(edge_energies.shape[0])
    assert np.max(uv_ids) == n_var - 1, str(np.max(uv_ids)) + " , " + str(n_var - 1)
    g = nifty.graph.UndirectedGraph(int(n_var))
    g.insertEdges(uv_ids)
    assert g.numberOfEdges == edge_energies.shape[0], "%i , %i" % (g.numberOfEdges, edge_energies.shape[0])
    assert g.numberOfEdges == uv_ids.shape[0], "%i, %i" % (g.numberOfEdges, uv_ids.shape[0])
    return nifty.graph.optimization.multicut.multicutObjective(g, edge_energies)


def multicut_exact(
        n_var,
        uv_ids,
        edge_energies,
        return_obj=False
):

    obj = nifty_objective(n_var, uv_ids, edge_energies)

    solver = obj.multicutIlpFactory(
        ilpSolver=ilp_bkend,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
    ).create(obj)

    t_inf = time.time()
    if ExperimentSettings().verbose:
        visitor = obj.verboseVisitor(1)
        ret = solver.optimize(visitor=visitor)
    else:
        ret = solver.optimize()
    t_inf = time.time() - t_inf

    mc_energy = obj.evalNodeLabels(ret)

    if not return_obj:
        return ret, mc_energy, t_inf
    else:
        return ret, mc_energy, t_inf, obj


def multicut_message_passing(
        n_var,
        uv_ids,
        edge_energies,
        n_threads=0,
        return_obj=False
):

    assert nifty.Configuration.WITH_LP_MP, "Message passing multicut needs nifty build with LP_MP"
    obj = nifty_objective(n_var, uv_ids, edge_energies)

    # TODO params params params
    solver = obj.multicutMpFactory(n_threads=n_threads).create(obj)

    t_inf = time.time()
    ret = solver.optimize()
    t_inf = time.time() - t_inf

    mc_energy = obj.evalNodeLabels(ret)

    if not return_obj:
        return ret, mc_energy, t_inf
    else:
        return ret, mc_energy, t_inf, obj


def multicut_fusionmoves(
        n_var,
        uv_ids,
        edge_energies,
        n_threads=0,
        return_obj=False
):

    obj = nifty_objective(n_var, uv_ids, edge_energies)

    # initialize the fusion moves solver with the greedy and
    # kernighan lin solution
    # greedy = obj.greedyAdditiveFactory()
    kl_factory = obj.multicutAndresKernighanLinFactory(greedyWarmstart=True)

    ilp_fac = obj.multicutIlpFactory(
        ilpSolver=ilp_bkend,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
    )

    fm_factory = obj.fusionMoveBasedFactory(
        verbose=1,
        fusionMove=obj.fusionMoveSettings(mcFactory=ilp_fac),
        proposalGen=obj.watershedProposals(sigma=10, seedFraction=ExperimentSettings().seed_fraction),
        numberOfIterations=ExperimentSettings().num_it,
        numberOfParallelProposals=2 * n_threads,
        numberOfThreads=n_threads,
        stopIfNoImprovement=ExperimentSettings().num_it_stop,
        fuseN=2,
    )

    factory = obj.chainedSolversFactory([kl_factory, fm_factory])
    solver = factory.create(obj)

    t_inf = time.time()
    if ExperimentSettings().verbose:
        visitor = obj.verboseVisitor(1)
        ret = solver.optimize(visitor=visitor)
    else:
        ret = solver.optimize()

    t_inf = time.time() - t_inf

    mc_energy = obj.evalNodeLabels(ret)
    if not return_obj:
        return ret, mc_energy, t_inf
    else:
        return ret, mc_energy, t_inf, obj


# needs opengm
def export_opengm_model(
        n_var,
        uv_ids,
        edge_energies,
        out_file):

    import opengm
    states = np.ones(n_var) * n_var
    gm_global = opengm.gm(states)
    # potts model
    potts_shape = [n_var, n_var]
    potts = opengm.pottsFunctions(potts_shape,
                                  np.zeros_like(edge_energies),
                                  edge_energies)
    # potts model to opengm function
    fids_b = gm_global.addFunctions(potts)
    gm_global.addFactors(fids_b, uv_ids.astype('uint32'))

    # save the opengm model
    opengm.saveGm(gm_global, out_file)
