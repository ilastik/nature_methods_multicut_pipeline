import numpy as np

from multicut_src import load_dataset, ExperimentSettings
from multicut_src import multicut_workflow
from init_test import meta_folder_anisotropic, meta_folder_isotropic

def test_mcwf_iso():
    pass

def test_mcwf_aniso(solver):
    assert solver in ("multicut_fusionmoves", "multicut_exact", "multicut_message_passing")
    ExperimentSettings().solver = solver

    ExperimentSettings().n_trees = 250
    ExperimentSettings().use_2d  = True
    ExperimentSettings().anisotropy_factor = 20
    ExperimentSettings().use_2rfs = True

    feat_list = ('raw', 'prob', 'reg')

    ds = load_dataset(meta_folder_anisotropic, 'test')
    mc_nodes, mc_edges, mc_energy, t_inf = multicut_workflow(ds, ds, 0, 0, feat_list)

    print "++++++++++++++++++++++++++++++++++"
    print "Multicut with solver %s succesful:" % solver
    print "MC-Energy: %f" % mc_energy
    print "T-inf: %f" % t_inf
    print "++++++++++++++++++++++++++++++++++"


if __name__ == '__main__':
    test_mcwf_aniso('multicut_fusionmoves')
