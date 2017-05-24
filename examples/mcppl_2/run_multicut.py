import vigra
import os
import argparse

from multicut_src import multicut_workflow  # , multicut_workflow_with_defect_correction
from multicut_src import ExperimentSettings, load_dataset


def run_mc(cache_folder, ds_train_name, ds_test_name, save_path):

    assert os.path.exists(os.path.split(save_path)[0]), "Please choose an existing folder to save your results"

    # if you have added multiple segmentations, you can choose on which one to run
    # experiments with the seg_id
    seg_id = 0

    # these strings encode the features that are used for the local features
    feature_list = ['raw', 'prob', 'reg']

    ds_train = load_dataset(cache_folder, ds_train_name)
    ds_test  = load_dataset(cache_folder, ds_test_name)

    # use this for running the mc without defected slices
    mc_nodes, _, _, _ = multicut_workflow(
        ds_train, ds_test,
        seg_id, seg_id,
        feature_list
    )

    # use this for running the mc with defected slices
    # mc_nodes, _, _, _ = multicut_workflow_with_defect_correction(
    #         ds_train, ds_test,
    #         seg_id, seg_id,
    #         feature_list)

    segmentation = ds_test.project_mc_result(seg_id, mc_nodes)
    vigra.writeHDF5(segmentation, save_path, 'data', compression='gzip')


def run_experiment(cache_folder):

    # this object stores different  experiment settings
    ExperimentSettings().rf_cache_folder = os.path.join(cache_folder, "rf_cache")

    # set to 1. for isotropic data,
    # to the actual degree for mildly anisotropic data
    # or to > 20. to compute filters in 2d
    ExperimentSettings().anisotropy_factor = 25.

    # set to true for segmentations with flat superpixels
    ExperimentSettings().use2d = True

    # number of threads used for multithreaded computations
    ExperimentSettings().n_threads = 8

    # number of trees used in the random forest
    ExperimentSettings().n_trees = 200

    # solver used for the multicut
    ExperimentSettings().solver  = "multicut_fusionmoves"
    ExperimentSettings().verbose = True

    # weighting scheme for edge-costs in the mc problem
    # set to 'none' for no weighting
    # 'z' or 'xyz' or 'all' for flat superpixels (z usually works best)
    # 'all' for isotropic data with 3d superpixel
    ExperimentSettings().weighting_scheme = "z"

    # path to save the segmentation result, order has to already exist
    save_path = os.path.join(cache_folder, 'mc_segmentation.h5')
    run_mc(cache_folder, 'my_train', 'my_test', save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_folder')
    args = parser.parse_args()
    cache_folder = args.cache_folder
    assert os.path.exists(cache_folder), cache_folder
    return cache_folder


if __name__ == '__main__':
    cache_folder = parse_args()
    run_experiment(cache_folder)
