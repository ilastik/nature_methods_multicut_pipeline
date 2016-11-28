# multicut only in 2d
# init isbi datasets
import os
import vigra
import numpy as np

from MCSolverImpl import multicut_exact, probs_to_energies
from ExperimentSettings import ExperimentSettings
from EdgeRF import learn_and_predict_rf_from_gt
from MetaSet import MetaSet
from DataSet import DataSet

from Tools import edges_to_binary
from eval_isbi import eval_lazy

# toplevel folder that has all the data

data_path = "/home/consti/Work/nature_experiments/isbi12_data"
meta_folder = "/home/consti/Work/nature_experiments/cache/isbi12_2d"

meta = MetaSet(meta_folder)

# init dataset for the isbi2012 train block
def init_isbi2012_train():
    isbi2012_train = DataSet(meta_folder, "isbi2012_train")

    raw_path = os.path.join(data_path, "raw/train-volume.h5")
    raw_key  = "data"
    # nasims baseline prob map
    inp_path = os.path.join(data_path, "probabilities/old_probs/nasims_oldbaseline_train.h5")
    inp_key  = "exported_data"

    isbi2012_train.add_raw(raw_path, raw_key)
    isbi2012_train.add_input(inp_path, inp_key)

    # 2d wsdt on namsis pmap
    seg_path0 = os.path.join(data_path, "watersheds/old_watersheds/ws_dt_nasims_baseline_train.h5")
    seg_key = "superpixel"

    isbi2012_train.add_seg(seg_path0, seg_key)

    # layerwise gt
    gt_path = os.path.join(data_path, "groundtruth/gt_cleaned.h5")
    isbi2012_train.add_gt(gt_path, "data")

    meta.add_dataset("isbi2012_train", isbi2012_train)


# init dataset for the isbi2012 test block
def init_isbi2012_test():
    isbi2012_test = DataSet(meta_folder, "isbi2012_test")

    raw_path = os.path.join(data_path, "raw/test-volume.h5")
    raw_key  = "data"

    inp_path = os.path.join(data_path, "probabilities/old_probs/nasims_oldbaseline_test.h5")
    inp_key  = "exported_data"

    isbi2012_test.add_raw(raw_path, raw_key)
    isbi2012_test.add_input(inp_path, inp_key)

    seg_key = "superpixel"

    seg_path0 = os.path.join(data_path, "watersheds/old_watersheds/ws_dt_nasims_baseline_test.h5")
    isbi2012_test.add_seg(seg_path0, seg_key)

    meta.add_dataset("isbi2012_test", isbi2012_test)


def mc_2d(ds_train_str, ds_test_str, local_feats, mc_params):
    meta.load()
    ds_train = meta.get_dataset(ds_train_str)
    ds_test = meta.get_dataset(ds_test_str)

    mcseg = np.zeros(ds_test.shape, dtype = np.uint32)
    rag = ds_test._rag(0)
    seg = ds_test.seg(0)

    # get edge probabilities from random forest
    edge_probs = learn_and_predict_rf_from_gt(mc_params.rf_cache_folder,
            ds_train, ds_test,
            0, 0,
            local_feats, mc_params)

    edge_energies = probs_to_energies(ds_test,
        edge_probs, 0, mc_params)

    # edges to slice
    slice_to_edge = {}
    for edge in rag.edgeIter():
        edge_coords = rag.edgeCoordinates(edge)
        z = edge_coords[0,2]
        if z - int(z) == 0:
            if z in slice_to_edge:
                slice_to_edge[z].append(edge.id)
            else:
                slice_to_edge[z] = [edge.id]

    edges = np.zeros(rag.edgeNum)
    for z in xrange(mcseg.shape[2]):

        energies_z = edge_energies[ slice_to_edge[z] ]

        seg_z = seg[:,:,z]
        rag_z = vigra.graphs.regionAdjacencyGraph( vigra.graphs.gridGraph(seg_z.shape[0:2] ),
                seg_z.astype(np.uint32) )
        edges_local_to_global = {}
        for node in rag_z.nodeIter():
            for adj_node in rag_z.neighbourNodeIter(node):
                edge_local = rag_z.findEdge(node, adj_node)
                edge_global = rag.findEdge(node, adj_node)
                edges_local_to_global[edge_local.id] = edge_global.id

        uvids_z = np.sort( rag_z.uvIds(), axis = 1 )
        nvar_z = uvids_z.max() + 1

        # TODO get uvids and nvar in z
        nodes_z, edges_z, _, _ = multicut_exact(
                nvar_z, uvids_z,
                energies_z, mc_params)

        for local_edge in edges_local_to_global:
            global_edge = edges_local_to_global[local_edge]
            edges[global_edge] = edges_z[local_edge]

    if ds_test_str == "isbi2012_train":
        ri, vi = eval_lazy(edges, rag)
        return ri, vi
    else:
        edge_vol = edges_to_binary( rag, edges )
        vigra.impex.writeVolume(edge_vol, "./mc2d_edges.tif", '', dtype = np.uint8 )
        return 0,0


if __name__ == '__main__':

    if True:
        init_isbi2012_train()
        init_isbi2012_test()
        meta.save()

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

    ri, vi = mc_2d("isbi2012_train", "isbi2012_train", local_feats_list, mc_params)
    mc_2d("isbi2012_train", "isbi2012_test", local_feats_list, mc_params)

    print "RI:", ri
    print "VI:", vi
