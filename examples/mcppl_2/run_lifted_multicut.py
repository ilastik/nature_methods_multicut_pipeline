import vigra
import os
import argparse

from multicut_src import lifted_multicut_workflow  # , lifted_multicut_workflow_with_defect_correction
from multicut_src import ExperimentSettings, load_dataset


def run_lmc(cache_folder, ds_train_name, ds_test_name):

    # path to save the segmentation result
    save_path = os.path.join(cache_folder, 'lmc_segmentation.h5')

    # if you have added multiple segmentations, you can choose on which one to run
    # experiments with the seg_id
    seg_id = 0

    # these strings encode the features that are used for the local features
    feature_list = ['raw', 'prob', 'reg']

    # these strings encode the features that will be used for the lifted edges
    feature_list_lifted = ['cluster', 'reg']

    # this factor determines the weighting of lifted vs. local edge costs
    gamma = 2.

    ds_train = load_dataset(cache_folder, ds_train_name)
    ds_test  = load_dataset(cache_folder, ds_test_name)

    # use this for running the mc without defected slices
    mc_nodes, _, _, _ = lifted_multicut_workflow(
        ds_train, ds_test,
        seg_id, seg_id,
        feature_list, feature_list_lifted,
        gamma=gamma
    )

    # use this for running the mc with defected slices
    # mc_nodes, _, _, _ = lifted_multicut_workflow_with_defect_correction(
    #     ds_train, ds_test,
    #     seg_id, seg_id,
    #     feature_list, feature_list_lifted,
    #     gamma=gamma)

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

    # use 2 different random forests for xy-and z edges
    # only makes sense for anisotropic data with flat superpixels
    ExperimentSettings().use_2rfs = True

    # solver used for the multicut
    ExperimentSettings().solver  = "multicut_fusionmoves"
    ExperimentSettings().verbose = True

    # weighting scheme for edge-costs in the mc problem
    # set to 'none' for no weighting
    # 'z' or 'xyz' or 'all' for flat superpixels (z usually works best)
    # 'all' for isotropic data with 3d superpixel
    ExperimentSettings().weighting_scheme = "z"

    # range of lifted edges
    ExperimentSettings().lifted_neighborhood = 3

    run_lmc(cache_folder, 'my_train', 'my_test')


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
