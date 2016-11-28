# init isbi datasets
import os

from MetaSet import MetaSet
from DataSet import DataSet

# toplevel folder that has all the data

# laptop
data_path = "/home/consti/Work/nature_experiments/isbi12_data"
# sirherny / fatchicken
#data_path = "/home/constantin/home_hdd/data/isbi12_data"

# cache folder
# laptop
meta_folder = "/home/consti/Work/nature_experiments/cache/isbi12"
# sirherny / fatchicken
#meta_folder = "/home/constantin/home_hdd/cache/isbi12"

meta = MetaSet(meta_folder)

# init dataset for the isbi2012 train block
def init_isbi2012_train():
    isbi2012_train = DataSet(meta_folder, "isbi2012_train")

    raw_path = os.path.join(data_path, "raw/train-volume.h5")
    raw_key  = "data"
    # nasims baseline prob map
    inp_path = os.path.join(data_path, "probabilities/unet_train.h5")
    inp_key  = "data"

    isbi2012_train.add_raw(raw_path, raw_key)
    isbi2012_train.add_input(inp_path, inp_key)

    # 2d wsdt on namsis pmap
    seg_path0 = os.path.join(data_path, "watersheds/wsdt_unet_train.h5")
    seg_path1 = os.path.join(data_path, "watersheds/wssmoothed_unet_train.h5")
    seg_key = "data"

    isbi2012_train.add_seg(seg_path0, seg_key)
    isbi2012_train.add_seg(seg_path1, seg_key)

    # layerwise gt
    gt_path = os.path.join(data_path, "groundtruth/gt_cleaned.h5")
    isbi2012_train.add_gt(gt_path, "data")

    # cutouts for learning the lifted multicut
    isbi2012_train.make_cutout([0,512,0,512,0,5])
    isbi2012_train.make_cutout([0,512,0,512,5,25])
    isbi2012_train.make_cutout([0,512,0,512,25,30])

    meta.add_dataset("isbi2012_train", isbi2012_train)


# init dataset for the isbi2012 test block
def init_isbi2012_test():
    isbi2012_test = DataSet(meta_folder, "isbi2012_test")

    raw_path = os.path.join(data_path, "raw/test-volume.h5")
    raw_key  = "data"

    inp_path = os.path.join(data_path, "probabilities/unet_test.h5")
    inp_key  = "data"

    isbi2012_test.add_raw(raw_path, raw_key)
    isbi2012_test.add_input(inp_path, inp_key)

    seg_path0 = os.path.join(data_path, "watersheds/wsdt_unet_test.h5")
    seg_path1 = os.path.join(data_path, "watersheds/wssmoothed_unet_test.h5")
    seg_key = "data"

    isbi2012_test.add_seg(seg_path0, seg_key)
    isbi2012_test.add_seg(seg_path1, seg_key)

    meta.add_dataset("isbi2012_test", isbi2012_test)


if __name__ == '__main__':
    init_isbi2012_train()
    init_isbi2012_test()
    meta.save()
