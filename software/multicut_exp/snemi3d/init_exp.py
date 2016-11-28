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

def init_testds():
    ds = DataSet("/home/consti/Work/data_neuro/cache/testds", "ds_test")

    raw_path = "/home/consti/Work/data_neuro/test_block/test-raw.h5"
    seg_path = "/home/consti/Work/data_neuro/test_block/test-seg.h5"
    prob_path = "/home/consti/Work/data_neuro/test_block/test-probs.h5"
    gt_path = "/home/consti/Work/data_neuro/test_block/test-gt.h5"

    ds.add_raw(raw_path, "data")
    ds.add_input(prob_path, "data")
    ds.add_seg(seg_path, "data")
    ds.add_gt(gt_path, "data")

    meta.add_dataset("ds_test", ds_test)

def init_snemi3d_train():
    snemi3d_train = DataSet(meta_folder, "snemi3d_train")

    raw_path = os.path.join(data_path, "raw/train-input.h5")
    raw_key  = "data"
    # path to the idsia probability maps
    inp_path = os.path.join(data_path, "probabilities/train-probs-nn.h5")
    inp_key  = "exported_data"

    snemi3d_train.add_raw(raw_path, raw_key)
    snemi3d_train.add_input(inp_path, inp_key)

    # 2d - distance trafo watershed on idsia pmaps
    seg_path0 = os.path.join(data_path, "watersheds/wsdt_2d_train.h5")
    # 2d distance trafo watershed on idsia pmaps with myelin segs inserted
    seg_path1 = os.path.join(data_path, "watersheds/myelin_train.h5")
    seg_key   = "superpixel"

    snemi3d_train.add_seg(seg_path0, seg_key)
    snemi3d_train.add_seg(seg_path1, seg_key)

    gt_path = os.path.join(data_path, "groundtruth/train-gt.h5")
    gt_key  = "gt"

    snemi3d_train.add_gt(gt_path, gt_key)

    # cutouts for LMC
    #snemi3d_train.make_cutout([0,1024,0,1024,0,20]) # lower 20 slices
    #snemi3d_train.make_cutout([0,1024,0,1024,20,80]) # upper 60 slices
    #snemi3d_train.make_cutout([0,1024,0,1024,80,100]) # upper 20 slices

    # train / validation split
    snemi3d_train.make_cutout([0,1024,0,1024,0,50]) # lower 20 slices
    snemi3d_train.make_cutout([0,1024,0,1024,50,100]) # upper 60 slices

    meta.add_dataset("snemi3d_train", snemi3d_train)


def init_snemi3d_test():
    snemi3d_test = DataSet(meta_folder, "snemi3d_test")

    raw_path = os.path.join(data_path, "raw/test-input.h5")
    raw_key  = "data"
    # path to the idsia probability maps
    inp_path = os.path.join(data_path, "probabilities/test-probs-nn.h5")
    # path to idsia prob maps refined with autocontext
    inp_key   = "exported_data"

    snemi3d_test.add_raw(raw_path, raw_key)
    snemi3d_test.add_input(inp_path, inp_key)

    # 2d - distance trafo watershed on idsia pmaps
    seg_path0 = os.path.join(data_path, "watersheds/wsdt_2d_test.h5")
    # 2d distance trafo watershed on idsia pmaps with myelin segs inserted
    seg_path1 = os.path.join(data_path, "watersheds/myelin_test.h5")
    seg_key   = "superpixel"

    snemi3d_test.add_seg(seg_path0, seg_key)
    snemi3d_test.add_seg(seg_path1, seg_key)

    meta.add_dataset("snemi3d_test", snemi3d_test)

if __name__ == '__main__':
    meta.load()
    init_testds()
    #init_snemi3d_train()
    #init_snemi3d_test()
    meta.save()
