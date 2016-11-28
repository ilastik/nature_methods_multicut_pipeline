import os

from MCSolver import lifted_multicut_workflow
from MetaSet import MetaSet
from ExperimentSettings import ExperimentSettings

from init_exp import meta


def snemi3d_lmc(ds_train_str, ds_test_str,
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
            local_feats_list, lifted_feats_list, mc_params)

    mc_seg = ds_test.project_mc_result(seg_id_test, mc_node)

    return mc_seg


def regfeats():
    meta.load()
    ds = meta.get_dataset("snemi3d_train")


if __name__ == '__main__':

    # parameters for the Multicut
    mc_params = ExperimentSettings()
    mc_params.set_rfcache(os.path.join(meta.meta_folder, "rf_cache"))
    mc_params.set_nthreads(20)

    mc_params.set_anisotropy(5.)
    mc_params.set_use2d(True)
    mc_params.set_ignore_mask(True)

    mc_params.set_ntrees(1000)
    mc_params.set_solver("opengm_fusionmoves")
    mc_params.set_verbose(False)
    mc_params.set_weighting_scheme("all")

    mc_params.set_lifted_neighborhood(3)

    local_feats_list = ("raw", "prob", "reg", "topo")
    lifted_feats_list = ("mc", "cluster", "reg")
    seg_id = 0

    mc_seg = snemi3d_lmc("snemi3d_train", "snemi3d_test",
            seg_id, seg_id,
            local_feats_list, lifted_feats_list, mc_params)

    vigra.writeHDF5(mc_seg, "mc_results.h5", "seg")
