# run init.py and run_multicuts.py before running this script

import os
import numpy as np
import cPickle as pickle
import vigra
from multicut_src import load_dataset
from multicut_src import compute_false_merges
from multicut_src import resolve_merges_with_lifted_edges
from multicut_src import RandomForest
from multicut_src import ExperimentSettings
import nifty_with_cplex.graph.rag as nrag


def detect_false_merges(
        ds_test_name,
        ds_train_name,
        meta_folder,
        test_seg_path, test_seg_key,
        train_seg_paths, train_seg_keys
):

    ExperimentSettings().anisotropy_factor = 1.
    ExperimentSettings().n_threads = 40
    ExperimentSettings().n_trees = 500
    ExperimentSettings().rf_cache_folder = rf_cache_folder

    # Set the path features that will be used
    ExperimentSettings().path_features = ['path_features',
                                          'lengths',
                                          'multicuts',
                                          'cut_features',
                                          'cut_features_whole_plane']
    # Within cut features
    ExperimentSettings().use_probs_map_for_cut_features = True

    ds_train = load_dataset(meta_folder, ds_train_name)
    ds_test = load_dataset(meta_folder, ds_test_name)

    # Path folders
    test_paths_cache_folder = os.path.join(meta_folder, ds_test_name, 'path_data')
    train_paths_cache_folder = os.path.join(meta_folder, 'train_path_data')

    # Detect false merges
    # Note that multiple train datasets and for each train dataset multiple segmentations can be added
    #   This also means that trainsets is an array of datasets of shape = (N, 1) where N is the number of datasets and
    #   the train segmentation paths have shape = (N, M) where M is the number of segmentations per trainset
    #   Also refer to the function description
    _, false_merge_probs, _ = compute_false_merges(
        [ds_train],  # Supply as list (see above)
        ds_test,
        [[train_seg_paths]], [[train_seg_keys]],  # 2D lists (see above)
        test_seg_path, test_seg_key,
        test_paths_cache_folder,
        train_paths_cache_folder
    )

    # This writes the prediction result into the cache folder.
    # The pickle file will be loaded by the next step
    with open(os.path.join(test_paths_cache_folder, 'false_paths_predictions.pkl'), 'w') as f:
        pickle.dump(false_merge_probs, f)


def resolve_false_merges(
        ds_test_name,
        ds_train_name,
        meta_folder, rf_cache_folder,
        new_nodes_filepath,
        pre_seg_filepath, pre_seg_key,
        min_prob_thresh, max_prob_thresh
):
    # Path folders
    paths_cache_folder = os.path.join(meta_folder, ds_test_name, 'path_data')

    # Load the trainset
    ds_train = load_dataset(meta_folder, ds_train_name)
    # Generate the rf cache name
    # TODO: This should be integrated into the pipeline
    rf_cache_name = 'rf_merges_%s' % '_'.join([ds.ds_name for ds in ds_train])

    # Load the testset
    ds_test = load_dataset(meta_folder, ds_test_name)
    seg_id = 0

    # Load the paths and the respective merge prediction to ultimately decide which objects to resolve
    path_data_filepath = os.path.join(paths_cache_folder, 'paths_ds_{}.h5'.format(ds_test_name))
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    if paths.size:
        paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(paths_cache_folder, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    # Find objects where probability >= min_prob_thresh and <= max_prob_thresh
    objs_with_prob_greater_thresh = np.unique(
        np.array(paths_to_objs)[
            np.logical_and(
                false_merge_probs >= min_prob_thresh,
                false_merge_probs <= max_prob_thresh
            )
        ]
    )

    # Extract all paths for each of the found objects
    false_paths = {}
    for obj in objs_with_prob_greater_thresh:
        false_paths[obj] = np.array(paths)[paths_to_objs == obj]

    # Load the previously generated path random forest.
    # This is needed as additional sampled paths are generated within the resolving step
    # which need to be classified as well
    rf_filepath = os.path.join(rf_cache_folder, rf_cache_name)
    path_rf = RandomForest.load_from_file(rf_filepath, 'rf', ExperimentSettings().n_threads)

    # This is the initial segmentation that is sought to be improved
    mc_segmentation = vigra.readHDF5(pre_seg_filepath, pre_seg_key)

    # Locations of multicut weights which are needed for the resolving step
    # Note that the filename can vary depending of the parameters used in the respective multicut
    # TODO: This should be integrated into the pipeline
    weight_filepath = os.path.join(
        meta_folder, ds_test_name,
        'probs_to_energies_0_{}_16.0_0.5_rawprobreg.h5'.format(ExperimentSettings().weighting_scheme)
    )
    lifted_filepath = os.path.join(
        meta_folder, ds_test_name,
        'lifted_probs_to_energies_0_3_0.5_2.0.h5'
    )
    mc_weights_all = vigra.readHDF5(weight_filepath, "data")
    lifted_weights_all = vigra.readHDF5(lifted_filepath, "data")

    # The resolving function
    new_node_labels = resolve_merges_with_lifted_edges(
        ds_test,
        [ds_train],  # Trainsets are a list of shape=(N, 1) -> see explanation in detect_merges and function description
        seg_id,
        false_paths,
        path_rf,
        mc_segmentation,
        mc_weights_all,
        paths_cache_folder=paths_cache_folder,  # If this is not supplied the paths will not be cached
        lifted_weights_all=lifted_weights_all  # This has to be supplied
    )

    # The graph-based result is stored
    # To project this to a segmentation see project_resolved_objects_to_segmentation
    with open(new_nodes_filepath, 'w') as f:
        pickle.dump(new_node_labels, f)


def project_resolved_objects_to_segmentation(
        meta_folder, ds_name,
        mc_seg_filepath, mc_seg_key,
        new_nodes_filepath,
        save_path, results_name
):
    ds = load_dataset(meta_folder, ds_name)
    seg_id = 0

    mc_segmentation = vigra.readHDF5(mc_seg_filepath, mc_seg_key)

    # Load resolving result
    with open(new_nodes_filepath) as f:
        resolved_objs = pickle.load(f)

    # Each object that was resolved is projected back to the segmentation individually
    rag = ds.rag(seg_id)
    mc_labeling = nrag.gridRagAccumulateLabels(rag, mc_segmentation)
    new_label_offset = np.max(mc_labeling) + 1
    for obj in resolved_objs:
        resolved_nodes = resolved_objs[obj]
        for node_id in resolved_nodes:
            mc_labeling[node_id] = new_label_offset + resolved_nodes[node_id]
        new_label_offset += np.max(resolved_nodes.values()) + 1
    mc_segmentation = nrag.projectScalarNodeDataToPixels(rag, mc_labeling, ExperimentSettings().n_threads)

    # Write the result
    vigra.writeHDF5(mc_segmentation, save_path, results_name, compression='gzip')


if __name__ == '__main__':
    from init_exp import meta_folder

    pre_seg_filepath = os.path.join(meta_folder, 'segmentation_test.h5')
    train_seg_filepath = os.path.join(meta_folder, 'segmentation_train.h5')

    # The merge detection step
    detect_false_merges(
        'test',
        'path_train',
        meta_folder,
        pre_seg_filepath, 'data',
        train_seg_filepath, 'data'
    )

    # Cache folder for all RF classifiers
    rf_cache_folder = os.path.join(meta_folder, 'rf_cache')
    new_nodes_filepath = os.path.join(meta_folder, 'test', 'new_ones_local.pkl')

    # The resolving step of those objects that have at least one path that was classified as containing a false merge.
    resolve_false_merges(
        'test',
        'path_train',
        meta_folder,
        rf_cache_folder,
        new_nodes_filepath,
        pre_seg_filepath, 'data',
        0.5, 1  # Defines lower and upper threshold for path probability representing a false merge
    )

    # The resolved result is projected back to a label image
    project_resolved_objects_to_segmentation(
        meta_folder,
        'test',
        pre_seg_filepath, 'data',
        new_nodes_filepath,
        os.path.join(meta_folder, 'segmentation_test_resolved.h5'), 'data'
    )