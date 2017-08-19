from __future__ import print_function, division
import os
import numpy as np
import h5py
import vigra

from .DataSet import Cutout, DataSet
from .ExperimentSettings import ExperimentSettings
from .lifted_mc import compute_and_save_lifted_nh, lifted_feature_aggregator, lifted_probs_to_energies
from .lifted_mc import lifted_hard_gt, mask_lifted_edges, optimize_lifted, learn_and_predict_lifted_rf
from .tools import find_matching_row_indices
from .EdgeRF import learn_and_predict_rf_from_gt, RandomForest
from .MCSolverImpl import probs_to_energies
from .MCSolver import _get_feat_str, run_mc_solver

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


# TODO cache
# get the long range z adjacency
def get_long_range_z_adjacency(ds, seg_id, long_range):
    rag = ds.rag(seg_id)
    adjacency = nrag.getLongRangeAdjacency(rag, long_range)
    if ds.has_seg_mask:
        where_uv = (adjacency != ExperimentSettings().ignore_seg_value).all(axis=1)
        adjacency = adjacency[where_uv]
    return adjacency


# get features of the long range z-adjacency from affinity maps
def get_long_range_z_features(ds, seg_id, affinity_map_path, affinity_map_key, long_range):

    # load the affinity maps from file
    with h5py.File(affinity_map_path, 'r') as f:
        affinity_maps = f[affinity_map_key]
        # if this is a cutout, we only load the relevant part of the affinity maps
        if isinstance(ds, Cutout):
            affinity_maps = affinity_maps[ds.bb]
        else:
            affinity_maps = affinity_maps[:]

    # some consistency checks
    # this assumes affinity channel as last channel
    assert affinity_maps.ndim == 4
    assert affinity_maps.shape[-1] == long_range
    assert affinity_maps.shape[:-1] == ds.shape

    rag = ds.rag(seg_id)
    adjacency = get_long_range_z_adjacency(ds, seg_id, long_range)

    long_range_feats = nrag.accumulateLongRangeFeatures(
        rag,
        affinity_maps,
        adjacency,
        numberOfThreads=ExperimentSettings().n_threads
    )
    return np.nan_to_num(long_range_feats)


# match the standard lifted neighborhood (lifted-nh) to the z-adjacency (lifted-range)
def match_to_lifted_nh(ds, seg_id, lifted_nh, long_range):
    assert lifted_nh >= long_range
    uv_lifted = compute_and_save_lifted_nh(ds, seg_id, lifted_nh)
    uv_long_range = get_long_range_z_adjacency(ds, seg_id, long_range)
    # match
    matches = find_matching_row_indices(uv_long_range, uv_lifted)[:, 0]
    assert len(matches) == len(uv_long_range)
    return matches


def learn_long_range_rf(
    trainsets,
    seg_id,
    feature_list_lifted,
    feature_list_local,
    affinity_map_paths,
    affinity_map_keys,
    trainstr,
    paramstr,
    with_defects=False
):
    cache_folder = ExperimentSettings().rf_cache_folder
    # check if already cached
    if cache_folder is not None:  # we use caching for the rf => look if already exists
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        rf_folder = os.path.join(cache_folder, "long_range_rf" + trainstr)
        rf_name = "rf_" + "_".join([trainstr, paramstr]) + ".h5"
        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)
        rf_path   = os.path.join(rf_folder, rf_name)
        if os.path.exists(rf_path):
            return RandomForest.load_from_file(rf_path, 'rf', ExperimentSettings().n_threads)

    features_train = []
    labels_train   = []

    for ii, ds_train in enumerate(trainsets):

        assert ds_train.n_cutouts == 3, "Wrong number of cutouts: " + str(ds_train.n_cutouts)
        train_cut = ds_train.get_cutout(1)

        # get edge probabilities from random forest on training set cut out in the middle
        p_local_train = learn_and_predict_rf_from_gt(
            [ds_train.get_cutout(0), ds_train.get_cutout(2)],
            train_cut,
            seg_id,
            seg_id,
            feature_list_local,
            with_defects=with_defects,
            use_2rfs=ExperimentSettings().use_2rfs
        )

        uv_ids_train = compute_and_save_lifted_nh(
            train_cut,
            seg_id,
            ExperimentSettings().lifted_neighborhood,
            with_defects)

        # compute the standard features for the training set
        f_train = lifted_feature_aggregator(
            train_cut,
            [ds_train.get_cutout(0), ds_train.get_cutout(2)],
            feature_list_lifted,
            feature_list_local,
            p_local_train,
            uv_ids_train,
            seg_id,
            with_defects)

        # get the labels and find the edges that should be used for training
        labels = lifted_hard_gt(train_cut, seg_id, uv_ids_train, with_defects)
        labeled = mask_lifted_edges(
            train_cut,
            seg_id,
            labels,
            uv_ids_train,
            with_defects
        )

        # get the long range features and keep only standard features / labels
        # that also have long range features
        long_range_feats = get_long_range_z_features(
            train_cut,
            seg_id,
            affinity_map_paths[ii],
            affinity_map_keys[ii],
            ExperimentSettings().long_range
        )

        to_lifted_nh = match_to_lifted_nh(
            train_cut,
            ExperimentSettings().lifted_neighborhood,
            ExperimentSettings().long_range
        )
        f_train = np.concatenate([f_train[labeled], long_range_feats], axis=1)
        labels, labeled = labels[to_lifted_nh], labeled[to_lifted_nh]

        features_train.append(f_train[labeled])
        labels_train.append(labels[labeled])

    features_train = np.concatenate(features_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)
    assert len(features_train) == len(labels_train)

    print("Start learning long range random forest")
    rf = RandomForest(
        features_train.astype('float32'),
        labels_train.astype('uint32'),
        n_trees=ExperimentSettings().n_trees,
        n_threads=ExperimentSettings().n_threads,
        max_depth=10
    )

    if cache_folder is not None:
        rf.write(rf_path, 'rf')
    return rf


# TODO
# learn lifted rf with long ange features
def learn_and_predict_long_range_rf(
    trainsets,
    ds_test,
    seg_id_train,
    seg_id_test,
    feature_list_lifted,
    feature_list_local,
    affinity_map_paths_train,
    affinity_map_keys_train,
    affinity_map_path_test,
    affinity_map_key_test,
    with_defects=False
):

    assert isinstance(trainsets, (DataSet, list, tuple)), type(trainsets)
    if not isinstance(trainsets, (list, tuple)):
        trainsets = [trainsets]

    assert isinstance(affinity_map_paths_train, (str, list, tuple)), type(affinity_map_paths_train)
    if not isinstance(affinity_map_paths_train, (list, tuple)):
        affinity_map_paths_train = [affinity_map_paths_train]

    assert isinstance(affinity_map_keys_train, (str, list, tuple)), type(affinity_map_keys_train)
    if not isinstance(affinity_map_keys_train, (list, tuple)):
        affinity_map_keys_train = [affinity_map_keys_train]

    assert len(affinity_map_paths_train) == len(trainsets)
    assert len(affinity_map_paths_train) == len(affinity_map_keys_train)

    # strings for caching
    # str for all relevant params
    paramstr = "_".join(
        ["_".join(feature_list_lifted), "_".join(feature_list_local),
         str(ExperimentSettings().anisotropy_factor), str(ExperimentSettings().learn_2d),
         str(ExperimentSettings().use_2d), str(ExperimentSettings().lifted_neighborhood),
         str(ExperimentSettings().long_range),
         str(ExperimentSettings().use_ignore_mask), str(with_defects)]
    )
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    trainstr = "_".join([ds.ds_name for ds in trainsets]) + "_" + str(seg_id_train)

    to_lifted_nh = match_to_lifted_nh(
        ds_test,
        seg_id_test,
        ExperimentSettings().lifted_neighborhood,
        ExperimentSettings().long_range
    )

    # check if rf is already cached, if we use caching for random forests ( == rf_cache folder is not None )
    # we cache predictions in the ds_train cache folder
    if ExperimentSettings().rf_cache_folder is not None:
        pred_name = "long_range_prediction_" + "_".join([trainstr, teststr, paramstr]) + ".h5"
        if len(pred_name) >= 255:
            pred_name = str(hash(pred_name[:-3])) + ".h5"
        pred_path = os.path.join(ds_test.cache_folder, pred_name)
        # see if the rf is already learned and predicted, otherwise learn it
        if os.path.exists(pred_path):
            return vigra.readHDF5(pred_path, 'data'), to_lifted_nh

    uv_ids_test = compute_and_save_lifted_nh(
        ds_test,
        seg_id_test,
        ExperimentSettings().lifted_neighborhood,
        with_defects
    )

    rf = learn_long_range_rf(
        trainsets,
        seg_id_train,
        feature_list_lifted,
        feature_list_local,
        affinity_map_paths_train,
        affinity_map_keys_train,
        trainstr,
        paramstr,
        with_defects
    )

    # get edge probabilities from random forest on test set
    p_local_test = learn_and_predict_rf_from_gt(
        [ds_train.get_cutout(i) for i in (0, 2) for ds_train in trainsets], ds_test,
        seg_id_train, seg_id_test,
        feature_list_local,
        with_defects=with_defects,
        use_2rfs=ExperimentSettings().use_2rfs
    )

    # get the normal lifted features + long range features and combine them
    features_test = lifted_feature_aggregator(
        ds_test,
        [ds_train.get_cutout(i) for i in (0, 2) for ds_train in trainsets],
        feature_list_lifted,
        feature_list_local,
        p_local_test,
        uv_ids_test,
        seg_id_test,
        with_defects
    )
    long_range_features = get_long_range_z_features(
        ds_test,
        seg_id_test,
        affinity_map_path_test,
        affinity_map_key_test,
        ExperimentSettings().long_range
    )
    features_test = np.concatenate([features_test[to_lifted_nh], long_range_features], axis=1)

    print("Start prediction long-range random forest")
    p_test = rf.predict_probabilities(features_test.astype('float32'))[:, 1]
    if ExperimentSettings().rf_cache_folder is not None:
        vigra.writeHDF5(p_test, pred_path, 'data')

    return p_test, to_lifted_nh


def long_range_multicut_wokflow(
    trainsets,
    ds_test,
    seg_id_train,
    seg_id_test,
    feature_list_local,
    feature_list_lifted,
    feature_list_long_range,
    affinity_map_paths_train,
    affinity_map_keys_train,
    affinity_map_path_test,
    affinity_map_key_test,
    gamma_lifted=2.,
    gamma_long_range=1.,
    warmstart=False,
    weight_z_lifted=False

):
    assert isinstance(ds_test, DataSet)
    assert isinstance(trainsets, (DataSet, list, tuple))

    print("Running long-range multicut on", ds_test.ds_name)
    if isinstance(trainsets, DataSet):
        print("Weights learned on", trainsets.ds_name)
    else:
        print("Weights learned on multiple datasets")

    # ) step one, train a random forest
    print("Start learning")

    # get the normal lifted prediction
    p_test_lifted, uv_ids_lifted = learn_and_predict_lifted_rf(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list_lifted,
        feature_list_local
    )

    # get the long range lifted prediction
    p_test_long_range, to_lifted_nh = learn_and_predict_long_range_rf(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list_long_range,
        feature_list_local,
        affinity_map_paths_train,
        affinity_map_keys_train,
        affinity_map_path_test,
        affinity_map_key_test
    )

    # get edge probabilities from random forest on the complete training set
    p_test_local = learn_and_predict_rf_from_gt(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list_local,
        with_defects=False,
        use_2rfs=ExperimentSettings().use_2rfs
    )

    # energies for the multicut
    costs_local = probs_to_energies(
        ds_test,
        p_test_local,
        seg_id_test,
        ExperimentSettings().weighting_scheme,
        ExperimentSettings().weight,
        ExperimentSettings().beta_local,
        _get_feat_str(feature_list_local)
    )

    # calculate the z distance for edges if 'weight_z_lifted == True'
    if weight_z_lifted:
        nz_test = ds_test.node_z_coord(seg_id_test)
        # node z to edge z distance
        edge_z_distance = np.abs(nz_test[uv_ids_lifted[:, 0]] - nz_test[uv_ids_lifted[:, 1]])
    else:
        edge_z_distance = None

    # get the costs for lifted and long range edges
    costs_lifted = lifted_probs_to_energies(
        ds_test,
        p_test_lifted,
        seg_id_test,
        edge_z_distance,
        ExperimentSettings().lifted_neighborhood,
        ExperimentSettings().beta_lifted,
        gamma_lifted
    )

    costs_long_range = lifted_probs_to_energies(
        ds_test,
        p_test_long_range,
        seg_id_test,
        edge_z_distance,
        ExperimentSettings().lifted_neighborhood,
        ExperimentSettings().beta_lifted,
        gamma_long_range
    )

    # weighting edges with their length for proper lifted to local scaling
    costs_local  /= costs_local.shape[0]
    costs_lifted /= costs_lifted.shape[0]
    costs_long_range /= costs_long_range.shape[0]

    # combine lifted and long range costs
    costs_lifted[to_lifted_nh] = costs_long_range

    uvs_local = ds_test.uv_ids(seg_id_test)
    # warmstart with multicut result
    if warmstart:
        n_var = uvs_local.max() + 1
        starting_point, _, _, _ = run_mc_solver(n_var, uvs_local, costs_local)
    else:
        starting_point = None

    node_labels, e_lifted, t_lifted = optimize_lifted(
        uvs_local,
        uv_ids_lifted,
        costs_local,
        costs_lifted,
        starting_point
    )

    edge_labels = node_labels[uvs_local[:, 0]] != node_labels[uvs_local[:, 1]]
    return node_labels, edge_labels, e_lifted, t_lifted
