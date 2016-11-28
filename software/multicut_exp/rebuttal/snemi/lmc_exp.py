import os

from MCSolver import lifted_multicut_workflow
from MetaSet import MetaSet
from ExperimentSettings import ExperimentSettings

import vigra

from init_exp import meta


def snemi3d_lmc(ds_train_str, ds_test_str,
        seg_id_train, seg_id_test,
        local_feats_list, lifted_feats_list, mc_params, gamma = 2.):

    meta.load()

    ds_train = meta.get_dataset(ds_train_str)
    ds_test = meta.get_dataset(ds_test_str)
    ds_train.make_filters(0,mc_params.anisotropy_factor)
    ds_test.make_filters(0,mc_params.anisotropy_factor)
    ds_train.make_filters(1,mc_params.anisotropy_factor)
    ds_test.make_filters(1,mc_params.anisotropy_factor)
    mc_node, mc_edges, mc_energy, t_inf = lifted_multicut_workflow(
            ds_train, ds_test,
            seg_id_train, seg_id_test,
            local_feats_list, lifted_feats_list, mc_params, gamma = gamma)

    mc_seg = ds_test.project_mc_result(seg_id_test, mc_node)

    return mc_seg


if __name__ == '__main__':

    # parameters for the Multicut
    mc_params = ExperimentSettings()
    mc_params.set_rfcache(os.path.join(meta.meta_folder, "rf_cache"))
    mc_params.set_nthreads(30)

    mc_params.set_anisotropy(5.)
    mc_params.set_use2d(True)
    mc_params.set_ignore_mask(True)

    mc_params.set_fuzzy_learning(True)
    mc_params.set_negative_threshold(0.4)
    mc_params.set_positive_threshold(0.6)

    mc_params.set_ntrees(800)
    mc_params.set_solver("nifty_fusionmoves")
    mc_params.set_verbose(False)

    mc_params.set_lifted_neighborhood(3)

    local_feats_list = ("prob", "reg", "topo")
    lifted_feats_list = ("cluster", "reg")
    seg_id = 0

    w = 'all'
    mc_params.set_weighting_scheme(w)


    lmc_seg = snemi3d_lmc("snemi3d_train", "snemi3d_train", seg_id, seg_id,
            local_feats_list, lifted_feats_list, mc_params, 2.)

    vigra.writeHDF5(lmc_seg, "snemi_final_seglmc_train.h5", "data", compression = 'gzip')
