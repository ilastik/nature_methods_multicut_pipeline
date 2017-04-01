import os
from multicut_src import MetaSet, Dataset

meta_folder = '/path/to/cachedir' # need path to directory for storing caches
if not os.path.exists(meta_folder):
    os.mkdir(meta_folder)
# metaset, caching all datasets that are used for the actual experiments
meta = MetaSet(meta_folder)


def init_trainset(train_name):
    meta.load()
    ds = DataSet(meta_folder, train_name) # init the dataset

    # filepaths to the input data
    raw_path = '/path/to/raw.h5' # path to raw data. For anisotropic data, must be in axis order x,y,z (with anisotropy in z)
    assert os.path.exists(raw_path)
    pmap_path = '/path/to/pmap.h5' # path to raw data. For anisotropic data, must be in axis order x,y,z
    assert os.path.exists(pmap_path)
    seg_path = '/path/to/seg.h5' # path to oversegmentation. For anisotropic data, must be in axis order x,y,z
    assert os.path.exists(seg_path)
    gt_path = '/path/to/gt.h5' # path to groundtruth. For anisotropic data, must be in axis order x,y,z
    assert os.path.exists(seg_path)

    # add the data from filepaths to the groundtruth
    # alternatively, you can also produce the inputs on the fly and add with '.add_*_from_data'
    ds.add_raw(raw_path, 'data') # here we assume that everything has the key 'data', change accordingly
    ds.add_input(pmap_path, 'data') # here we assume that everything has the key 'data', change accordingly
    ds.add_seg(seg_path, 'data') # here we assume that everything has the key 'data', change accordingly
    ds.add_gt(gt_path, 'data') # here we assume that everything has the key 'data', change accordingly

    # in addition, you can make cutouts here that can be used e.g. for validation

    # add the dataset to the metaset
    meta.add_dataset(train_name, ds)
    meta.save()


def init_testset(test_name):
    meta.load()
    ds = DataSet(meta_folder, test_name) # init the dataset

    # filepaths to the input data
    raw_path = '/path/to/raw.h5' # path to raw data. For anisotropic data, must be in axis order x,y,z (with anisotropy in z)
    assert os.path.exists(raw_path)
    pmap_path = '/path/to/pmap.h5' # path to raw data. For anisotropic data, must be in axis order x,y,z
    assert os.path.exists(pmap_path)
    seg_path = '/path/to/seg.h5' # path to oversegmentation. For anisotropic data, must be in axis order x,y,z
    assert os.path.exists(seg_path)

    # add the data from filepaths to the groundtruth
    # alternatively, you can also produce the inputs on the fly and add with '.add_*_from_data'
    ds.add_raw(raw_path, 'data') # here we assume that everything has the key 'data', change accordingly
    ds.add_input(pmap_path, 'data') # here we assume that everything has the key 'data', change accordingly
    ds.add_seg(seg_path, 'data') # here we assume that everything has the key 'data', change accordingly

    # in addition, you can make cutouts here that can be used e.g. for validation

    # add the dataset to the metaset
    meta.add_dataset(test_name, ds)
    meta.save()


if __name__ == '__main__':
    init_trainset('my_train') # name / key of your dataset with training data (== having gt)
    init_testset('my_test')   # name / key of your dataset with test data
