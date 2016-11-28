# initialize the datasets for the snemi experiments

import os

from MetaSet import MetaSet
from DataSet import DataSet

# notebook
#meta_folder = "/home/consti/Work/data_neuro/cache/snemi3d"
#data_path = "/home/consti/Work/data_neuro/nature_experiments/snemi3d_data"

# sirherny / fatchicken
meta_folder = "/home/constantin/Work/home_hdd/cache/snemi3d"
data_path = "/home/constantin/Work/neurodata_hdd/snemi3d_data"

meta = MetaSet(meta_folder)

def init_snemi3d_train(for_validation = False):
    snemi3d_train = DataSet(meta_folder, "snemi3d_train")

    raw_path = os.path.join(data_path, "raw/train-input.h5")
    raw_key  = "data"
    # path to the idsia probability maps
    inp_path0 = os.path.join(data_path, "probabilities/SnemiTheMapTrain.h5")
    inp_key  = "data"

    snemi3d_train.add_raw(raw_path, raw_key)
    snemi3d_train.add_input(inp_path0, inp_key)
    #snemi3d_train.add_input(inp_path1, inp_key)

    # 2d - distance trafo watershed on idsia pmaps
    seg_path0 = os.path.join(data_path, "watersheds/snemiTheUltimateMapWsdtSpecialTrain_myel.h5")

    seg_key   = "data"

    snemi3d_train.add_seg(seg_path0, seg_key)

    gt_path = os.path.join(data_path, "groundtruth/train-gt.h5")
    gt_key  = "data"

    snemi3d_train.add_gt(gt_path, gt_key)

    if for_validation:
        # train / validation split
        snemi3d_train.make_cutout([0,1024,0,1024,0,50])
        snemi3d_train.make_cutout([0,1024,0,1024,50,100])

    else:
        # cutouts for LMC
        snemi3d_train.make_cutout([0,1024,0,1024,0,20]) # lower 20 slices
        snemi3d_train.make_cutout([0,1024,0,1024,20,80]) # upper 60 slices
        snemi3d_train.make_cutout([0,1024,0,1024,80,100]) # upper 20 slices

    meta.add_dataset("snemi3d_train", snemi3d_train)


def init_snemi3d_test():
    snemi3d_test = DataSet(meta_folder, "snemi3d_test")

    raw_path = os.path.join(data_path, "raw/test-input.h5")
    raw_key  = "data"
    # path to the idsia probability maps
    inp_path0 = os.path.join(data_path, "probabilities/SnemiTheUltimateMapTest.h5")
    # path to idsia prob maps refined with autocontext
    inp_key   = "data"

    snemi3d_test.add_raw(raw_path, raw_key)
    snemi3d_test.add_input(inp_path0, inp_key)

    # 2d - distance trafo watershed on idsia pmaps
    seg_path0 = os.path.join(data_path, "watersheds/snemiTheUltimateMapWsdtSpecialTest_myel.h5")
    seg_key   = "data"

    snemi3d_test.add_seg(seg_path0, seg_key)

    meta.add_dataset("snemi3d_test", snemi3d_test)


if __name__ == '__main__':
    #init_snemi3d_train()

    meta.load()
    init_snemi3d_train()
    meta.save()
