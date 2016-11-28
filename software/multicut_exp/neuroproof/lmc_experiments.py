import os

from MCSolver import lifted_multicut_workflow
from MetaSet import MetaSet
from ExperimentSettings import ExperimentSettings

from init_exp import meta


def neuroproof_lmc(ds_train_str, ds_test_str,
        seg_id_train, seg_id_test,
        local_feats_list, lifted_feats_list, mc_params):

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
            local_feats_list, lifted_feats_list, mc_params,
            gamma = 2., warmstart = False, weight_z_lifted = False )

    mc_seg = ds_test.project_mc_result(seg_id_test, mc_node)

    return mc_seg


if __name__ == '__main__':

    # parameters for the Multicut
    mc_params = ExperimentSettings()
    mc_params.set_rfcache(os.path.join(meta.meta_folder, "rf_cache"))
    mc_params.set_nthreads(15)

    mc_params.set_anisotropy(1.)
    mc_params.set_use2d(False)
    mc_params.set_ignore_mask(False)

    mc_params.set_fuzzy_learning(True)
    mc_params.set_negative_threshold(0.4)
    mc_params.set_positive_threshold(0.6)
    mc_params.lifted_neighborhood = 2

    mc_params.set_ntrees(600)
    #mc_params.set_weighting_scheme("all")

    mc_params.set_solver("nifty_exact")
    mc_params.set_verbose(True)

    local_feats_list = ("raw", "prob", "reg", "topo")
    lifted_feats_list = ("reg", "cluster")
    seg_id = 0

    mc_seg = neuroproof_lmc("neuroproof_train", "neuroproof_test",
            seg_id, seg_id,
            local_feats_list, lifted_feats_list,
            mc_params)

    vigra.writeHDF5(mc_seg, "/home/constantin/Work/home_hdd/results/nature_results/rebuttal/neuroproof/res_lmc.h5", "data", compression = 'gzip')
