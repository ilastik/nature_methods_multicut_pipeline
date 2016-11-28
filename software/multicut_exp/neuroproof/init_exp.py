# initialize the datasets for the snemi experiments

import os

from wsdt import wsDtSegmentation

from MetaSet import MetaSet
from DataSet import DataSet

# notebook
#meta_folder = "/home/consti/Work/nature_experiments/cache/neuroproof"
#data_path = "/home/consti/Work/nature_experiments/neuroproof_data"

# sirherny / fatchicken
meta_folder = "/home/constantin/Work/home_hdd/cache/neuroproof"
data_path = "/home/constantin/Work/neurodata_hdd/neuroproof_data/"

meta = MetaSet(meta_folder)

def make_ws(inp, thresh, sig_min):
    return wsDtSegmentation(inp, thresh, 25, 50, sig_min, 0., True)

def init_neuroproof_train():
    neuroproof_train = DataSet(meta_folder, "neuroproof_train")

    raw_path = os.path.join(data_path, "raw_train.h5")
    raw_key  = "data"

    inp_path = os.path.join(data_path, "probabilities_train.h5")
    inp_key  = "data"

    seg_path = os.path.join(data_path, "overseg_train.h5")
    seg_key   = "data"

    gt_path = os.path.join(data_path, "gt_train.h5")
    gt_key  = "data"

    assert os.path.exists(inp_path), inp_path
    assert os.path.exists(gt_path), gt_path

    neuroproof_train.add_raw(raw_path, raw_key)
    neuroproof_train.add_input(inp_path, inp_key)

    #seg = make_ws(neuroproof_train.inp(1), .3, 2.6)
    #neuroproof_train.add_seg_from_data(seg)
    neuroproof_train.add_seg(seg_path, seg_key)

    neuroproof_train.add_gt(gt_path, gt_key)

    neuroproof_train.make_cutout([0,250,0,250,0,50]) # lower 20 slices
    neuroproof_train.make_cutout([0,250,0,250,50,200]) # upper 60 slices
    neuroproof_train.make_cutout([0,250,0,250,200,250]) # upper 20 slices

    meta.add_dataset("neuroproof_train", neuroproof_train)


def init_neuroproof_test():
    neuroproof_test = DataSet(meta_folder, "neuroproof_test")

    raw_path = os.path.join(data_path, "raw_test.h5")
    raw_key  = "data"

    inp_path = os.path.join(data_path, "probabilities_test.h5")
    inp_key  = "data"

    seg_path  = os.path.join(data_path, "overseg_test.h5")
    seg_key   = "data"

    gt_path = os.path.join(data_path, "gt_test.h5")
    gt_key  = "data"

    neuroproof_test.add_raw(raw_path, raw_key)
    neuroproof_test.add_input(inp_path, inp_key)

    #seg = make_ws(neuroproof_test.inp(1), .3, 2.6)
    #neuroproof_test.add_seg_from_data(seg)
    neuroproof_test.add_seg(seg_path, seg_key)

    meta.add_dataset("neuroproof_test", neuroproof_test)


if __name__ == '__main__':
    init_neuroproof_train()
    init_neuroproof_test()
    meta.save()
