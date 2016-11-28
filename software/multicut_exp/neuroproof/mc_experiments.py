import os
import vigra
import numpy as np

from MCSolver import multicut_workflow
from EdgeRF import learn_and_predict_rf_from_gt
from MetaSet import MetaSet
from ExperimentSettings import ExperimentSettings

from init_exp import meta

def neuroproof_mc(ds_train_str, ds_test_str,
        seg_id_train, seg_id_test,
        local_feats_list, mc_params):

    meta.load()

    ds_train = meta.get_dataset(ds_train_str)
    ds_test = meta.get_dataset(ds_test_str)

    mc_node, mc_edges, mc_energy, t_inf = multicut_workflow(
            ds_train, ds_test,
            seg_id_train, seg_id_test,
            local_feats_list, mc_params)

    print np.unique(mc_node).shape

    mc_seg = ds_test.project_mc_result(seg_id_test, mc_node)

    return mc_seg


if __name__ == '__main__':

    # parameters for the Multicut
    mc_params = ExperimentSettings()
    mc_params.set_rfcache(os.path.join(meta.meta_folder, "rf_cache"))
    mc_params.set_nthreads(30)

    mc_params.set_anisotropy(1.)
    mc_params.set_use2d(False)
    mc_params.set_ignore_mask(False)

    mc_params.set_fuzzy_learning(True)
    mc_params.set_negative_threshold(0.4)
    mc_params.set_positive_threshold(0.6)

    mc_params.set_ntrees(750)
    #mc_params.set_weighting_scheme("all")

    mc_params.set_solver("opengm_exact")
    mc_params.set_verbose(True)

    local_feats_list = ("raw", "prob", "reg", "topo")
    seg_id = 0

    meta.load()

    mc_seg = neuroproof_mc("neuroproof_train", "neuroproof_test",
            seg_id, seg_id,
            local_feats_list, mc_params)

    vigra.writeHDF5(mc_seg, "/home/constantin/Work/home_hdd/results/nature_results/rebuttal/neuroproof/res_mc.h5", "data", compression = 'gzip')
