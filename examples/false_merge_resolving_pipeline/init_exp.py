
from multicut_src import DataSet


def init_dataset(
        meta_folder, name,
        raw_filepath, raw_name,
        probs_filepath, probs_name,
        seg_filepath, seg_name,
        gt_filepath=None, gt_name=None,
        make_cutouts=False
):

    # Also see init_exp.py in example mcppl_2

    # Init the dataset
    ds = DataSet(meta_folder, name)

    # Add data
    ds.add_raw(raw_filepath, raw_name)
    ds.add_input(probs_filepath, probs_name)
    ds.add_seg(seg_filepath, seg_name)
    if gt_filepath is not None:
        ds.add_gt(gt_filepath, gt_name)

    # add cutouts for lifted multicut training
    if make_cutouts:
        shape = ds.shape
        z_offset = 10
        ds.make_cutout([0, 0, 0], [shape[0], shape[1], z_offset])
        ds.make_cutout([0, 0, z_offset], [shape[0], shape[1], shape[2] - z_offset])
        ds.make_cutout([0, 0, shape[2] - z_offset], [shape[0], shape[1], shape[2]])


# This is the overall cache folder
meta_folder = 'path/to/cache/folder/'

if __name__ == '__main__':

    # Init a training dataset for (lifted) multicut training
    init_dataset(
        meta_folder, 'mc_train',
        'path/to/raw/data/mc_train_raw.h5', 'data',
        'path/to/membrane/probabilities/mc_train_probs.h5', 'data',
        'path/to/superpixels/mc_train_superpixels.h5', 'data',
        gt_filepath='path/to/gt/mc_train_gt.h5', gt_name='data',
        make_cutouts=True
    )

    # Init a training dataset for path classification training
    init_dataset(
        meta_folder, 'path_train',
        'path/to/raw/data/path_train_raw.h5', 'data',
        'path/to/membrane/probabilities/path_train_probs.h5', 'data',
        'path/to/superpixels/path_train_superpixels.h5', 'data',
        gt_filepath='path/to/gt/path_train_gt.h5', gt_name='data',
        make_cutouts=False
    )

    # Init a test dataset
    init_dataset(
        meta_folder, 'test',
        'path/to/raw/data/test_raw.h5', 'data',
        'path/to/membrane/probabilities/test_probs.h5', 'data',
        'path/to/superpixels/test_superpixels.h5', 'data',
        gt_filepath='path/to/gt/test_gt.h5', gt_name='data',
        make_cutouts=False
    )