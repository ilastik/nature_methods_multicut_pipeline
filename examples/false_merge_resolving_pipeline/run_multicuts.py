
# run init_exp.py before running this script

import os
import vigra

from multicut_src import multicut_workflow, lifted_multicut_workflow, load_dataset
from multicut_src import merge_small_segments
from multicut_src import ExperimentSettings


def run_mc(
        meta_folder,
        ds_train_name,
        ds_test_name,
        save_path,
        results_name,
):
    assert os.path.exists(os.path.split(save_path)[0]), "Please choose an existing folder to save your results"

    seg_id = 0

    feature_list = ['raw', 'prob', 'reg']

    ds_train = load_dataset(meta_folder, ds_train_name)
    ds_test = load_dataset(meta_folder, ds_test_name)

    mc_nodes, _, _, _ = multicut_workflow(
        ds_train,
        ds_test,
        seg_id, seg_id,
        feature_list
    )

    segmentation = ds_test.project_mc_result(seg_id, mc_nodes)

    # Merge small segments
    segmentation = merge_small_segments(segmentation.astype('uint32'), 100)

    # Store the final result
    vigra.writeHDF5(segmentation, save_path, results_name, compression = 'gzip')


def run_lifted_mc(
        meta_folder,
        ds_train_name,
        ds_test_name,
        save_path,
        results_name
):
    assert os.path.exists(os.path.split(save_path)[0]), "Please choose an existing folder to save your results"

    seg_id = 0

    feature_list = ['raw', 'prob', 'reg']
    feature_list_lifted = ['cluster', 'reg']

    gamma = 2.

    ds_train = load_dataset(meta_folder, ds_train_name)
    ds_test = load_dataset(meta_folder, ds_test_name)

    mc_nodes, _, _, _ = lifted_multicut_workflow(
        ds_train, ds_test,
        seg_id, seg_id,
        feature_list, feature_list_lifted,
        gamma=gamma
    )

    segmentation = ds_test.project_mc_result(seg_id, mc_nodes)

    # Merge small segments
    segmentation = merge_small_segments(segmentation.astype('uint32'), 100)

    # Store the final result
    vigra.writeHDF5(segmentation, save_path, results_name, compression = 'gzip')


def run_mc_for_path_train(meta_folder, rf_cache_folder):

    # Set the experiment settings
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 1.
    ExperimentSettings().use_2d = False
    ExperimentSettings().use_2rfs = False
    ExperimentSettings().n_threads = 32
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().weighting_scheme = 'all'

    run_mc(
        meta_folder,
        'mc_train',
        'path_train',
        os.path.join(meta_folder, 'segmentation_path_train.h5'),
        'data'
    )


def run_lmc_for_test_dataset(meta_folder, rf_cache_folder):

    # Set the experiment settings
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 1.
    ExperimentSettings().use_2d = False
    ExperimentSettings().use_2rfs = False
    ExperimentSettings().n_threads = 32
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().weighting_scheme = 'all'
    ExperimentSettings().lifted_neighborhood = 3

    run_lifted_mc(
        meta_folder,
        'mc_train',
        'test',
        os.path.join(meta_folder, 'segmentation_test.h5'),
        'data'
    )


if __name__ == '__main__':

    # Note: The settings in this example are for isotropic data

    # The meta folder is already defined in init_exp.py
    from init_exp import meta_folder

    # Cache folder for all RF classifiers
    rf_cache_folder = os.path.join(meta_folder, 'rf_cache')

    # Compute a multicut segmentation for the path train volume
    # In this example we compute a multicut to obtain a segmentation of lower quality than obtained by lifted multicut.
    # This way more merges will be present in the segmentation which leads to a higher amount of training instances.
    run_mc_for_path_train(meta_folder, rf_cache_folder)

    # Compute a lifted multicut segmentation for the test volume
    run_lmc_for_test_dataset(meta_folder, rf_cache_folder)