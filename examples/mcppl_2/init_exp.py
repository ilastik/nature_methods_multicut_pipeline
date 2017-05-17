import os
from multicut_src import Dataset

meta_folder = '/path/to/cachedir' # need path to directory for storing caches
if not os.path.exists(meta_folder):
    os.mkdir(meta_folder)


def init_trainset(train_name):

    ds = DataSet(meta_folder, train_name) # init the dataset

    # filepaths to the input data
    raw_path = '/path/to/raw.h5' # path to raw data. For anisotropic data, must be in axis order z,y,x (with anisotropy in z)
    assert os.path.exists(raw_path)
    pmap_path = '/path/to/pmap.h5' # path to raw data. For anisotropic data, must be in axis order z,y,x
    assert os.path.exists(pmap_path)
    seg_path = '/path/to/seg.h5' # path to oversegmentation. For anisotropic data, must be in axis order z,y,x
    assert os.path.exists(seg_path)
    gt_path = '/path/to/gt.h5' # path to groundtruth. For anisotropic data, must be in axis order z,y,x
    assert os.path.exists(gt_path)

    # in addition you can add a mask for parts of the segmentation that should not be learned / infered on:
    #mask_path = '/path/to/mask.h5'
    #assert os.path.exists(mask_path)

    # add the data from filepaths to the groundtruth
    # alternatively, you can also produce the inputs on the fly and add with '.add_*_from_data'
    ds.add_raw(raw_path, 'data') # here we assume that everything has the key 'data', change accordingly

    # you can also add a list of slices with defects, that will be treated differently, only makes sense for flat superpixel
    #ds.add_defect_slices([3,19,33]) # insert meaningful slice numbers...

    ds.add_input(pmap_path, 'data') # here we assume that everything has the key 'data', change accordingly

    #ds.add_seg_mask(mask_path, 'data') # THIS HAS TO BE ADDED BEFORE THE SEGMENTATION TO BE USED!
    ds.add_seg(seg_path, 'data') # here we assume that everything has the key 'data', change accordingly

    ds.add_gt(gt_path, 'data') # here we assume that everything has the key 'data', change accordingly

    # in addition, you can make cutouts here that can be used e.g. for validation
    # here, we add three cutouts that are used during the lifted multicut training
    z0 = 0
    z1 = 15
    z2 = shape[0] - 15
    z3 = shape[0]
    shape = ds.shape
    ds.make_cutout( [z0,0,0], [z1,shape[1],shape[2]] )
    ds.make_cutout( [z1,0,0], [z2,shape[1],shape[2]] )
    ds.make_cutout( [z2,0,0], [z3,shape[1],shape[2]] )


def init_testset(test_name):

    ds = DataSet(meta_folder, test_name) # init the dataset

    # filepaths to the input data
    raw_path = '/path/to/raw.h5' # path to raw data. For anisotropic data, must be in axis order z,y,x (with anisotropy in z)
    assert os.path.exists(raw_path)
    pmap_path = '/path/to/pmap.h5' # path to raw data. For anisotropic data, must be in axis order z,y,x
    assert os.path.exists(pmap_path)
    seg_path = '/path/to/seg.h5' # path to oversegmentation. For anisotropic data, must be in axis order z,y,x
    assert os.path.exists(seg_path)

    # in addition you can add a mask for parts of the segmentation that should not be learned / infered on:
    #mask_path = '/path/to/mask.h5'
    #assert os.path.exists(mask_path)

    # add the data from filepaths to the groundtruth
    # alternatively, you can also produce the inputs on the fly and add with '.add_*_from_data'
    ds.add_raw(raw_path, 'data') # here we assume that everything has the key 'data', change accordingly

    # you can also add a list of slices with defects, that will be treated differently, only makes sense for flat superpixel
    #ds.add_defect_slices([3,19,33]) # insert meaningful slice numbers...

    ds.add_input(pmap_path, 'data') # here we assume that everything has the key 'data', change accordingly

    #ds.add_seg_mask(mask_path, 'data') # THIS HAS TO BE ADDED BEFORE THE SEGMENTATION TO BE USED!
    ds.add_seg(seg_path, 'data') # here we assume that everything has the key 'data', change accordingly

    # in addition, you can make cutouts here that can be used e.g. for validation


if __name__ == '__main__':
    init_trainset('my_train') # name / key of your dataset with training data (== having gt)
    init_testset('my_test')   # name / key of your dataset with test data
