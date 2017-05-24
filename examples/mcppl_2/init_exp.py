import os
import argparse
from multicut_src import DataSet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_folder', type=str)
    parser.add_argument('test_data_folder', type=str)
    parser.add_argument('cache_folder', type=str)
    args = parser.parse_args()

    train_data_folder = args.train_data_folder
    assert os.path.exists(train_data_folder), train_data_folder

    test_data_folder = args.test_data_folder
    assert os.path.exists(test_data_folder), test_data_folder

    cache_folder = args.cache_folder
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    return train_data_folder, test_data_folder, cache_folder


def print_instructions(e):
    print "Input files not found."
    print "Either download and unzip the test data from:"
    print "https://drive.google.com/open?id=0B4_sYa95eLJ1ek8yMWozTzhBbGM"
    print "or provide your own data"
    print "(check init_trainset / init_testset for the input format)"
    print "Failed with assertion error:"
    raise e


def init_trainset(data_folder, cache_folder, train_name):

    ds = DataSet(cache_folder, train_name)  # init the dataset

    # filepaths to the input data

    # path to raw data. For anisotropic data, must be in axis order z,y,x (with anisotropy in z)
    raw_path = os.path.join(data_folder, 'raw.h5')
    assert os.path.exists(raw_path), raw_path

    # path to probability maps. For anisotropic data, must be in axis order z,y,x
    pmap_path = os.path.join(data_folder, 'pmap.h5')
    assert os.path.exists(pmap_path), pmap_path

    # path to oversegmentation. For anisotropic data, must be in axis order z,y,x
    seg_path = os.path.join(data_folder, 'seg.h5')
    assert os.path.exists(seg_path), seg_path

    # path to groundtruth. For anisotropic data, must be in axis order z,y,x
    gt_path = os.path.join(data_folder, 'gt.h5')
    assert os.path.exists(gt_path), gt_path

    # in addition you can add a mask for parts of the segmentation that should not be learned / infered on:
    # mask_path = '/path/to/mask.h5'
    # assert os.path.exists(mask_path)

    # you can also add a list of slices with defects, that will be treated differently,
    # only makes sense for anisotropic data with flat superpixel
    # ds.add_defect_slices([3,19,33]) # insert meaningful slice numbers...

    # add the data from filepaths to the training dataset

    # alternatively, you can also produce the inputs on the fly and add with '.add_*_from_data'
    # here we assume that everything has the key 'data', change accordingly
    ds.add_raw(raw_path, 'data')
    ds.add_input(pmap_path, 'data')
    ds.add_seg(seg_path, 'data')
    ds.add_gt(gt_path, 'data')

    # add the seg mask if present
    # ds.add_seg_mask(mask_path, 'data')

    # in addition, you can make cutouts here that can be used e.g. for validation
    # here, we add three cutouts that are used during the lifted multicut training
    shape = ds.shape
    z0 = 0
    z1 = 15
    z2 = shape[0] - 15
    z3 = shape[0]
    shape = ds.shape
    ds.make_cutout([z0, 0, 0], [z1, shape[1], shape[2]])
    ds.make_cutout([z1, 0, 0], [z2, shape[1], shape[2]])
    ds.make_cutout([z2, 0, 0], [z3, shape[1], shape[2]])


def init_testset(data_folder, cache_folder, test_name):

    ds = DataSet(cache_folder, test_name)  # init the dataset

    # filepaths to the input data

    # path to raw data. For anisotropic data, must be in axis order z,y,x (with anisotropy in z)
    raw_path = os.path.join(data_folder, 'raw.h5')
    assert os.path.exists(raw_path)

    # path to probability maps. For anisotropic data, must be in axis order z,y,x
    pmap_path = os.path.join(data_folder, 'pmap.h5')
    assert os.path.exists(pmap_path)

    # path to oversegmentation. For anisotropic data, must be in axis order z,y,x
    seg_path = os.path.join(data_folder, 'seg.h5')
    assert os.path.exists(seg_path)

    # in addition you can add a mask for parts of the segmentation that should not be learned / infered on:
    # mask_path = '/path/to/mask.h5'
    # assert os.path.exists(mask_path)

    # you can also add a list of slices with defects, that will be treated differently,
    # only makes sense for anisotropic data with flat superpixel
    # ds.add_defect_slices([3,19,33]) # insert meaningful slice numbers...

    # add the data from filepaths to the test dataset

    # alternatively, you can also produce the inputs on the fly and add with '.add_*_from_data'
    # here we assume that everything has the key 'data', change accordingly
    ds.add_raw(raw_path, 'data')
    ds.add_input(pmap_path, 'data')
    ds.add_seg(seg_path, 'data')

    # add the seg mask if present
    # ds.add_seg_mask(mask_path, 'data')

    # in addition, you can make cutouts here that can be used e.g. for validation


if __name__ == '__main__':
    train_data_folder, test_data_folder, cache_folder = parse_args()
    #try:
    # name / key of your dataset with training data (== having gt)
    init_trainset(train_data_folder, cache_folder, 'my_train')
    # name / key of your dataset with test data
    init_testset(test_data_folder, cache_folder, 'my_test')
    #except AssertionError as e:
    #    print_instructions(e)
