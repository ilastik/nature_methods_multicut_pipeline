import os
import vigra

from MCSolver import multicut_workflow
from MetaSet import MetaSet
from ExperimentSettings import ExperimentSettings

from init_exp import meta


def snemi3d_mc(ds_train_str, ds_test_str,
        seg_id_train, seg_id_test,
        local_feats_list, mc_params):

    meta.load()

    ds_train = meta.get_dataset(ds_train_str)
    ds_test = meta.get_dataset(ds_test_str)

    mc_node, mc_edges, mc_energy, t_inf = multicut_workflow(
            ds_train, ds_test,
            seg_id_train, seg_id_test,
            local_feats_list, mc_params)

    mc_seg = ds_test.project_mc_result(seg_id_test, mc_node)

    print mc_energy, t_inf

    return mc_seg


if __name__ == '__main__':

    # parameters for the Multicut
    mc_params = ExperimentSettings()
    mc_params.set_rfcache(os.path.join(meta.meta_folder, "rf_cache"))
    mc_params.set_nthreads(20)

    mc_params.set_anisotropy(5.)
    mc_params.set_use2d(True)
    mc_params.set_ignore_mask(True)

    mc_params.set_ntrees(1000)
    mc_params.set_weighting_scheme("all")

    mc_params.set_seed_fraction(0.05)
    mc_params.set_solver("opengm_exact")

    mc_params.set_verbose(True)

    local_feats_list = ("raw", "prob", "reg", "topo")
    seg_id = 0

    mc_seg = snemi3d_mc("snemi3d_train", "snemi3d_test",
            seg_id, seg_id,
            local_feats_list, mc_params)

    vigra.writeHDF5(mc_seg, "snemi_result_train.h5", "data")
