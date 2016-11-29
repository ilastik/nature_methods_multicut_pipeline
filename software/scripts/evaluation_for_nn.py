import vigra
import numpy as np
import os

from shutil import rmtree

from wsdt import wsDtSegmentation

from MetaSet import MetaSet
from DataSet import DataSet
from MCSolver import multicut_workflow
from ExperimentSettings import ExperimentSettings


cache_folder = ""

meta = MetaSet(cache_folder)

def watersheds(prob_map):

    threshold = 0.25
    # off the shelve settings
    minMemSize = 15
    minSegSize = 30
    sigMinima  = 2.0
    sigWeights = 2.6
    groupSeeds = True

    segmentation = np.zeros_like(prob_map, dtype = np.uint32)
    offset = 0
    for z in xrange(prob_map.shape[2]):

        wsdt = wsDtSegmentation(prob_map[:,:,z], threshold,
            minMemSize, minSegSize,
            sigMinima, sigWeights, groupSeeds)
        segmentation[:,:,z] = wsdt
        segmentation[:,:,z] += offset
        offset = np.max(segmentation)

    return segmentation


def init_A(path_to_nn):
    path_to_raw = ""
    key_to_raw  = ""

    key_to_nn   = ""

    path_to_gt  = ""
    key_to_gt   = ""

    if os.path.exists( os.path.join(cache_folder, "sampleA") ):

        meta.load()
        rmtree(os.path.join(cache_folder, "sampleA"))

        sampleA = DataSet(cache_folder, "sampleA")

        sampleA.add_raw(path_to_raw, key_to_raw)
        sampleA.add_input(path_to_nn, key_to_nn)

        sampleA.add_seg_from_data(watersheds(sampleA.inp(1)))

        sampleA.add_gt(path_to_gt, key_to_gt)

        sampleA.make_cutout([0,1250,0,1250,0,35])
        sampleA.make_cutout([0,1250,0,1250,35,75])

        meta.update_dataset("sampleA", sampleA)
        meta.save()

    else:

        sampleA = DataSet(cache_folder, "sampleA")

        sampleA.add_raw(path_to_raw, key_to_raw)
        sampleA.add_input(path_to_nn, key_to_nn)

        sampleA.add_seg_from_data(watersheds(sampleA.inp(1)))

        sampleA.add_gt(path_to_gt, key_to_gt)

        sampleA.make_cutout([0,1250,0,1250,0,35])
        sampleA.make_cutout([0,1250,0,1250,35,75])

        meta.add_dataset("sampleA", sampleA)
        meta.save()


def init_B(path_to_nn):
    path_to_raw = ""
    key_to_raw  = ""

    key_to_nn   = ""

    path_to_gt  = ""
    key_to_gt   = ""

    if os.path.exists( os.path.join(cache_folder, "sampleB") ):

        meta.load()
        rmtree(os.path.join(cache_folder, "sampleB"))

        sampleB = DataSet(cache_folder, "sampleB")

        sampleB.add_raw(path_to_raw, key_to_raw)
        sampleB.add_input(path_to_nn, key_to_nn)

        sampleB.add_seg_from_data(watersheds(sampleB.inp(1)))

        sampleB.add_gt(path_to_gt, key_to_gt)

        sampleB.make_cutout([0,1250,0,1250,0,35])
        sampleB.make_cutout([0,1250,0,1250,35,75])

        meta.update_dataset("sampleB", sampleB)
        meta.save()

    else:

        sampleB = DataSet(cache_folder, "sampleB")

        sampleB.add_raw(path_to_raw, key_to_raw)
        sampleB.add_input(path_to_nn, key_to_nn)

        sampleB.add_seg_from_data(watersheds(sampleB.inp(1)))

        sampleB.add_gt(path_to_gt, key_to_gt)

        sampleB.make_cutout([0,1250,0,1250,0,35])
        sampleB.make_cutout([0,1250,0,1250,35,75])

        meta.add_dataset("sampleB", sampleB)
        meta.save()


def init_C(path_to_nn):
    path_to_raw = ""
    key_to_raw  = ""

    key_to_nn   = ""

    path_to_gt  = ""
    key_to_gt   = ""

    if os.path.exists( os.path.join(cache_folder, "sampleC") ):
        meta.load()

        rmtree(os.path.join(cache_folder, "sampleC"))

        sampleC = DataSet(cache_folder, "sampleC")

        sampleC.add_raw(path_to_raw, key_to_raw)
        sampleC.add_input(path_to_nn, key_to_nn)

        sampleC.add_seg_from_data(watersheds(sampleC.inp(1)))

        sampleC.add_gt(path_to_gt, key_to_gt)

        sampleC.make_cutout([0,1250,0,1250,0,35])
        sampleC.make_cutout([0,1250,0,1250,35,75])

        meta.update_dataset("sampleC", sampleC)
        meta.save()

    else:

        sampleC = DataSet(cache_folder, "sampleC")

        sampleC.add_raw(path_to_raw, key_to_raw)
        sampleC.add_input(path_to_nn, key_to_nn)

        sampleC.add_seg_from_data(watersheds(sampleC.inp(1)))

        sampleC.add_gt(path_to_gt, key_to_gt)

        sampleC.make_cutout([0,1250,0,1250,0,35])
        sampleC.make_cutout([0,1250,0,1250,35,75])

        meta.add_dataset("sampleC", sampleC)
        meta.save()



def run_and_eval_mc(sample):

    # Experiment settings
    exp_params = ExperimentSettings()
    exp_params.set_ntrees(300)
    exp_params.set_anisotropy(25.)
    exp_params.set_weighting_scheme("z")
    exp_params.set_solver("opengm_fusionmoves")

    local_feats_list  = ("prob", "reg")
    sample.make_filters(1, 25.)

    mc_node, mc_edges, mc_energy, t_inf = multicut_workflow(
        sample.get_cutout(0), sample.get_cutout(1),
        0,0,
        local_feats_list, exp_params)

    mc_seg = sample.get_cutout(1).project_mc_result(0, mc_node)
    print "MCSEG"

    # TODO eval



def eval_A():
    meta.load()
    run_and_eval_mc(meta.get_dataset("sampleA"))

def eval_B():
    meta.load()
    run_and_eval_mc(meta.get_dataset("sampleA"))

def eval_C():
    meta.load()
    run_and_eval_mc(meta.get_dataset("sampleA"))
