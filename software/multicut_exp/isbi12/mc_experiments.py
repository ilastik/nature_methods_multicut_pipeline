import os
import vigra
import numpy as np

from MCSolver import multicut_workflow
from ExperimentSettings import ExperimentSettings

from eval_isbi import eval_lazy
from Tools import edges_to_binary

from init_exp import meta

def isbi12_multicut(ds_train_str, ds_test_str,
        seg_id_train, seg_id_test,
        local_feats_list, mc_params):

    meta.load()
    ds_train = meta.get_dataset(ds_train_str)
    ds_test = meta.get_dataset(ds_test_str)

    mc_node, mc_edges, mc_energy, t_inf = multicut_workflow(
            ds_train, ds_test,
            seg_id_train, seg_id_test,
            local_feats_list, mc_params)

    if ds_test_str == "isbi2012_train":
        return eval_lazy(mc_edges, ds_test._rag(seg_id_test) )

    else:
        assert ds_test_str == "isbi2012_test"
        res_folder = "/home/consti/Work/nature_experiments/results/isbi12"
        mc_seg = ds_test.project_mc_result( seg_id_test, mc_node )

        # save segmentation result
        seg_name = "_".join( ["mcresult", str(seg_id_test), "seg"] ) + ".h5"
        seg_path = os.path.join(res_folder, seg_name)
        vigra.writeHDF5(mc_seg, seg_path, "data")

        # save binary edges
        edge_name = "_".join( ["mcresult", str(seg_id_test), "edges"] ) + ".tif"
        edge_path = os.path.join(res_folder, edge_name)
        edge_vol = edges_to_binary(ds_test._rag(seg_id_test), mc_edges)
        vigra.impex.writeVolume(edge_vol, edge_path, '', dtype = np.uint8 )
        return 0, 0


if __name__ == '__main__':

    # parameters for the Multicut
    mc_params = ExperimentSettings()

    # set the Random Forest Cache Folder
    mc_params.set_rfcache( os.path.join(meta.meta_folder,"rf_cache") )
    mc_params.set_nthreads(8)

    # parameters for features
    mc_params.set_anisotropy(25.)
    mc_params.set_use2d(True)

    # parameters for learning
    mc_params.set_fuzzy_learning(True)
    mc_params.set_negative_threshold(0.4)
    mc_params.set_positive_threshold(0.6)
    mc_params.set_learn2d(True)
    mc_params.set_ntrees(1000)
    mc_params.set_ignore_mask(False)

    # parameters for weighting
    mc_params.set_weighting_scheme("z")

    local_feats_list = ("raw", "prob", "reg", "topo")

    ris = {}
    vis = {}

    # 0: wsdt
    # 1: ws on smoothed probs
    for seg_id in (0,1):
        ri, vi = isbi12_multicut("isbi2012_train", "isbi2012_train",
                seg_id, seg_id,
                local_feats_list,
                mc_params)
        isbi12_multicut("isbi2012_train", "isbi2012_test",
                seg_id, seg_id,
                local_feats_list,
                mc_params)

        ris[seg_id] = ri
        vis[seg_id] = vi

    print "Train Scores:"
    for seg_id in (0,1):
        print "Seg_id:", seg_id
        print "RI:", ris[seg_id]
        print "VI:", vis[seg_id]
