import numpy as np
import vigra
import h5py
import os
from functools import partial

from DataSet import DataSet
from defect_handling import modified_edge_features, modified_region_features, modified_topology_features, modified_edge_features_from_affinity_maps
from defect_handling import modified_edge_indications, modified_edge_gt
from defect_handling import get_skip_edges, modified_adjacency, get_skip_ranges, get_skip_starts, get_ignore_edge_ids
from ExperimentSettings import ExperimentSettings
from tools import edges_to_volume, edges_to_volume_from_uvs_in_plane, edges_to_volume_from_uvs_between_plane, edges_to_volumes_for_skip_edges

RandomForest = vigra.learning.RandomForest3

# toplevel convenience function for features
# aggregates all the features given in feature list:
# possible valus: "raw" -> edge features from raw_data
# "prob" -> edge features from probability maps
# "reg"  -> features from region statistics
# "topo" -> topological features
def local_feature_aggregator(ds,
        seg_id,
        feature_list,
        anisotropy_factor = 1.,
        use_2d = False):

    assert seg_id < ds.n_seg, str(seg_id) + " , " + str(ds.n_seg)
    assert anisotropy_factor >= 1., "Finer resolution in z-direction is not supported"
    for feat in feature_list:
        assert feat in ("raw", "prob", "affinities", "extra_input", "reg", "topo"), feat
    features = []
    if "raw" in feature_list:
        features.append(
                ds.edge_features(seg_id, 0, anisotropy_factor ))
    if "prob" in feature_list:
        features.append(
                ds.edge_features(seg_id, 1, anisotropy_factor ))
    if "affinities" in feature_list:
        features.append(
                ds.edge_features_from_affinity_maps(seg_id, (1,2), anisotropy_factor ))
    if "extra_input" in feature_list:
        features.append(
                ds.edge_features(seg_id, 2, anisotropy_factor ))
    if "reg" in feature_list:
        features.append(
                ds.region_features(seg_id, 0,
            ds._adjacent_segments(seg_id), False ) )
    if "topo" in feature_list:
        features.append(
                ds.topology_features(seg_id, use_2d ))
    #if "curve" in feature_list:
    #    features.append(ds.curvature_features(seg_id))

    return np.concatenate(features, axis = 1)

# toplevel convenience function for features
# aggregates all the features given in feature list:
# possible valus: "raw" -> edge features from raw_data
# "prob" -> edge features from probability maps
# "reg"  -> features from region statistics
# "topo" -> topological features
def local_feature_aggregator_with_defects(ds,
        seg_id,
        feature_list,
        anisotropy_factor = 1.,
        use_2d = False):

    assert seg_id < ds.n_seg, str(seg_id) + " , " + str(ds.n_seg)
    assert anisotropy_factor >= 1., "Finer resolution in z-direction is not supported"
    for feat in feature_list:
        assert feat in ("raw", "prob", "affinities", "extra_input", "reg", "topo"), feat
    features = []
    if "raw" in feature_list:
        features.append(modified_edge_features(
            ds, seg_id,
            0, anisotropy_factor))
    if "prob" in feature_list:
        features.append(modified_edge_features(ds, seg_id,
            1, anisotropy_factor))
    if "affinities" in feature_list:
        features.append(modified_edge_features_from_affinity_maps(ds, seg_id,
            (1,2), anisotropy_factor))
    if "extra_input" in feature_list:
        features.append(modified_edge_features(ds, seg_id,
            2, anisotropy_factor))
    if "reg" in feature_list:
        features.append(modified_region_features(ds, seg_id,
            0, ds._adjacent_segments(seg_id), False) )
    if "topo" in feature_list:
        features.append(modified_topology_features(ds, seg_id, use_2d))
    #if "curve" in feature_list:
    #    features.append(ds.curvature_features(seg_id))

    return np.concatenate(features, axis = 1)


# edge masking:
# we set all labels that are going to be ignored to 0.5
def mask_edges(ds,
        seg_id,
        labels,
        exp_params,
        uv_ids,
        with_defects):

    labeled = np.ones_like(labels, dtype = bool)
    # set ignore mask to 0.5
    if exp_params.use_ignore_mask: # ignore mask not yet supported for defect pipeline
        ignore_mask = ds.ignore_mask(seg_id, uv_ids, with_defects)
        assert ignore_mask.shape[0] == labels.shape[0]
        labeled[ ignore_mask ] = False

    # ignore all edges that are connected to the ignore segment label in the seg mask
    if ds.has_seg_mask:
        ignore_mask = (uv_ids == ds.ignore_seg_value).any(axis = 1)
        assert ignore_mask.shape[0] == labels.shape[0]
        labeled[ ignore_mask ] = False

    # set z edges to 0.5
    if exp_params.learn_2d:
        edge_indications = modified_edge_indications(ds, seg_id) if with_defects else ds.edge_indications(seg_id)
        labeled[edge_indications == 0] = False

    # mask the ignore edges
    if with_defects and ds.has_defects:
        ignore_edge_ids = get_ignore_edge_ids(ds, seg_id)
        labeled[ignore_edge_ids] = False

    return labeled


# TODO make volumina viewer conda package and ship this too
def view_edges(ds, seg_id, uv_ids, labels, labeled, with_defects = False):
    assert uv_ids.shape[0] == labels.shape[0]
    assert labels.shape[0] == labeled.shape[0]
    from volumina_viewer import volumina_n_layer

    labels_debug = np.zeros(labels.shape, dtype = np.uint32)
    labels_debug[labels == 1.]  = 1
    labels_debug[labels == 0.]  = 2
    labels_debug[np.logical_not(labeled)] = 5

    if with_defects:
        skip_transition = labels_debug.shape[0] - get_skip_edges(ds, seg_id).shape[0]
        edge_indications = modified_edge_indications(ds, seg_id)[:skip_transition]
        # get  uv ids and labels for the skip edges
        uv_skip     = uv_ids[skip_transition:]
        labels_skip = labels_debug[skip_transition:]
        #get uv ids and labels for the normal edges
        uv_ids      = uv_ids[:skip_transition]
        labels_debug = labels_debug[:skip_transition]
    else:
        edge_indications = ds.edge_indications(seg_id)

    assert edge_indications.shape[0] == labels_debug.shape[0], "%i, %i" % (edge_indications.shape[0], labels_debug.shape[0])
    # xy - and z - labels
    labels_xy = labels_debug[edge_indications == 1]
    labels_z  = labels_debug[edge_indications == 0]
    uv_xy = uv_ids[edge_indications == 1]
    uv_z  = uv_ids[edge_indications == 0]

    seg = ds.seg(seg_id)
    edge_vol_xy   = edges_to_volume_from_uvs_in_plane(ds, seg, uv_xy, labels_xy)
    edge_vol_z_dn = edges_to_volume_from_uvs_between_plane(ds, seg, uv_z, labels_z, True)
    edge_vol_z_up = edges_to_volume_from_uvs_between_plane(ds, seg, uv_z, labels_z, False)

    raw = ds.inp(0).astype('float32')
    gt  = ds.gt()

    if with_defects:
        skip_ranges = get_skip_ranges(ds, seg_id)
        skip_starts = get_skip_starts(ds, seg_id)
        edge_vol_skip = edges_to_volumes_for_skip_edges(
                ds,
                seg,
                uv_skip,
                labels_skip,
                skip_starts,
                skip_ranges)
        volumina_n_layer(
                [raw, seg, gt, edge_vol_z_dn, edge_vol_z_up, edge_vol_skip, edge_vol_xy],
                ['raw', 'seg', 'groundtruth', 'labels_z_down', 'labels_z_up', 'labels_skip', 'labels_xy'])
    else:
        volumina_n_layer(
                [raw, seg, gt, edge_vol_z_dn, edge_vol_z_up, edge_vol_xy],
                ['raw', 'seg', 'groundtruth', 'labels_z_down', 'labels_z_up', 'labels_xy',])


# FIXME: For now, we don't support different feature strings for the edge types
# if features should differ for the edge types (e.g. affinities), these need to be
# already merged in the feature computation
def _learn_seperate_rfs(trainsets,
        seg_id,
        features,
        labels,
        labeled,
        exp_params,
        rf_path,
        features_skip = None,
        labels_skip = None,
        with_defects = False):

    assert len(trainsets) == len(features)
    assert len(labels) == len(features)
    assert len(labels) == len(labeled), "%i, %i" % (len(labels), len(labeled))

    if with_defects:
        skip_transitions = [features[i].shape[0] - get_skip_edges(ds, seg_id).shape[0] for i, ds in enumerate(trainsets)]

    all_indications = [modified_edge_indications(ds, seg_id)[labeled[i]][skip_transitions[i]:] \
            if (with_defects and ds.has_defects) else ds.edge_indications(seg_id)[labeled[i]] for i, ds in enumerate(trainsets)]

    features_xy = np.concatenate(
            [features[i][indications == 1] for i, indications in enumerate(all_indications)])
    labels_xy = np.concatenate(
            [labels[i][indications == 1] for i, indications in enumerate(all_indications)])

    features_z = np.concatenate(
            [features[i][indications == 0] for i, indications in enumerate(all_indications)])
    labels_z = np.concatenate(
            [labels[i][indications == 0] for i, indications in enumerate(all_indications)])

    assert features_xy.shape[0] == labels_xy.shape[0]
    assert features_z.shape[0] == labels_z.shape[0]
    assert all( np.unique(labels_xy) == np.array([0, 1]) ), "unique labels: " + str(np.unique(labels_xy))
    assert all( np.unique(labels_z) == np.array([0, 1]) ), "unique labels: " + str(np.unique(labels_z))

    print "Start learning random forest for xy edges"
    rf_xy = RandomForest(features_xy.astype('float32'), labels_xy,
        treeCount = exp_params.n_trees,
        n_threads = exp_params.n_threads)

    print "Start learning random forest for z edges"
    rf_z = RandomForest(features_z.astype('float32'), labels_z,
        treeCount = exp_params.n_trees,
        n_threads = exp_params.n_threads)

    if features_skip is not None:
        assert labels_skip is not None
        print "Start learning defect random forest"
        rf_defects = RandomForest(features_skip.astype('float32'), labels_skip,
            treeCount = exp_params.n_trees,
            n_threads = exp_params.n_threads)

    if rf_path is not None:
        rf_xy.writeHDF5(rf_path, 'rf_xy')
        rf_z.writeHDF5(rf_path, 'rf_z')
        if features_skip is not None:
            rf_defects.writeHDF5(rf_path, 'rf_defects')

    if features_skip is not None:
        return [rf_xy, rf_z, rf_defects]
    else:
        return [rf_xy, rf_z]


def _learn_single_rfs(
        features,
        labels,
        exp_params,
        rf_path,
        features_skip = None,
        labels_skip = None):

    features = np.concatenate(features)
    labels  = np.concatenate(labels)

    assert features.shape[0] == labels.shape[0]
    assert all( np.unique(labels) == np.array([0, 1]) ), "unique labels: " + str(np.unique(labels))

    print "Start learning random forest"
    rf = RandomForest(features.astype('float32'), labels,
        treeCount = exp_params.n_trees,
        n_threads = exp_params.n_threads)

    if features_skip is not None:
        assert labels_skip is not None
        print "Start learning defect random forest"
        rf_defects = RandomForest(features_skip.astype('float32'), labels_skip,
            treeCount = exp_params.n_trees,
            n_threads = exp_params.n_threads)

    if rf_path is not None:
        rf.writeHDF5(rf_path, 'rf')
        if with_defects:
            rf_defects.writeHDF5(rf_path, 'rf_defects')

    if features_skip is not None:
        return [rf, rf_defects]
    else:
        return [rf]


def learn_rf(cache_folder,
        trainsets,
        seg_id,
        feature_aggregator,
        exp_params,
        trainstr,
        paramstr,
        with_defects = False,
        use_2rfs = False):

    if cache_folder is not None: # we use caching for the rf => look if already exists
        rf_folder = os.path.join(cache_folder, "rf_" + trainstr)
        rf_name = "rf_" + "_".join( [trainstr, paramstr] ) + ".h5"
        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)
        rf_path   = os.path.join(rf_folder, rf_name)
        if os.path.exists(rf_path):
            print "Loading random forest from"
            print rf_path
            if use_2rfs:
                rfs = [RandomForest(rf_path, 'rf_xy'), RandomForest(rf_path, 'rf_z') ]
            else:
                rfs = [RandomForest(rf_path, 'rf') ]
            # we need to check if the defect rf actually exists
            with h5py.File(rf_path) as f:
                 has_defect_rf = True if ('rf_defects' in f.keys()) else False
            if has_defect_rf:
                assert with_defects
                rfs.append( RandomForest(rf_path, 'rf_defects') )
            return rfs

    features_train = []
    labels_train   = []
    labeled_train  = []

    if with_defects:
        features_skip = []
        labels_skip   = []

    for cutout in trainsets:

        assert isinstance(cutout, DataSet)
        assert cutout.has_gt

        features_cut = feature_aggregator( cutout, seg_id )

        if with_defects and cutout.has_defects:
            uv_ids = modified_adjacency(cutout, seg_id)
        else:
            uv_ids = cutout._adjacent_segments(seg_id)

        if exp_params.learn_fuzzy:
            # TODO implement for defects
            if with_defects:
                raise AttributeError("Fuzzy learning not supported for defect pipeline yet")
            labels_cut = cutout.edge_gt_fuzzy(seg_id,
                    exp_params.positive_threshold, exp_params.negative_threshold)
        else:
            labels_cut = modified_edge_gt(
                    cutout,
                    seg_id) if with_defects else cutout.edge_gt(seg_id)

        assert labels_cut.shape[0] == features_cut.shape[0], "%i, %i" % (labels_cut.shape[0], features_cut.shape[0])

        labeled = mask_edges(cutout,
                seg_id,
                labels_cut,
                exp_params,
                uv_ids,
                with_defects)
        labeled_train.append(labeled)

        # inspect the edges FIXME this has dependencies outside of conda, so we can't expose it for now
        # TODO properly inspect the skip edges
        if False:
            view_edges(cutout,
                    seg_id,
                    uv_ids,
                    labels_cut,
                    labeled,
                    with_defects and cutout.has_defects)

        features_cut = features_cut[labeled]
        labels_cut   = labels_cut[labeled].astype('uint32')

        if with_defects and cutout.has_defects:
            skip_transition = features_cut.shape[0] - get_skip_edges(cutout, seg_id).shape[0]
            features_skip.append(features_cut[skip_transition:])
            labels_skip.append(labels_cut[skip_transition:])
            features_cut = features_cut[:skip_transition]
            labels_cut = labels_cut[:skip_transition]

        features_train.append(features_cut)
        labels_train.append(labels_cut)

    if with_defects:
        if features_skip: # check if any features / labels for skip edges were added
            assert labels_skip
            features_skip = np.concatenate(features_skip)
            labels_skip = np.concatenate(labels_skip)
        else: # if not set with_defects to false, beacause we can't learn a defect rf
            with_defects = False

    # TODO move this to learn_single_rf / learn_seperate_rfs
    if use_2rfs:
        return _learn_seperate_rfs(trainsets, seg_id,
                features_train, labels_train,
                labeled_train, exp_params,
                rf_path if cache_folder is not None else None,
                features_skip if with_defects else None,
                labels_skip if with_defects else None,
                with_defects
                )
    else:
        return _learn_single_rfs(features_train, labels_train,
                exp_params,
                rf_path if cache_folder is not None else None,
                features_skip if with_defects else None,
                labels_skip if with_defects else None
                )


# set cache folder to None if you dont want to cache the resulting rf
# TODO use cacher hdf5 for caching!
def learn_and_predict_rf_from_gt(cache_folder,
        trainsets, ds_test,
        seg_id_train, seg_id_test,
        feature_list, exp_params,
        with_defects = False,
        use_2rfs     = False
        ):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)
    assert isinstance(ds_test, DataSet)
    assert isinstance(exp_params, ExperimentSettings)

    # for only a single ds, put it in a list
    if isinstance(trainsets, DataSet):
        trainsets = [trainsets]

    if with_defects:
        feature_aggregator = partial( local_feature_aggregator_with_defects,
            feature_list = feature_list,
            anisotropy_factor = exp_params.anisotropy_factor,
            use_2d = exp_params.use_2d )
    else:
        feature_aggregator = partial( local_feature_aggregator,
            feature_list = feature_list, anisotropy_factor = exp_params.anisotropy_factor,
            use_2d = exp_params.use_2d )

    # strings for caching
    # str for all relevant params
    paramstr = "_".join( ["_".join(feature_list), str(exp_params.anisotropy_factor),
        str(exp_params.learn_2d), str(exp_params.learn_fuzzy),
        str(exp_params.n_trees), str(exp_params.negative_threshold),
        str(exp_params.positive_threshold), str(exp_params.use_2d),
        str(exp_params.use_ignore_mask), str(with_defects), str(use_2rfs)] )
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    trainstr = "_".join([ds.ds_name for ds in trainsets ]) + "_" + str(seg_id_train)

    if cache_folder is not None: # cache-folder exists => look if we already have a prediction

        pred_folder = os.path.join(cache_folder, "pred_" + trainstr)
        pred_name = "prediction_" + "_".join([trainstr, teststr, paramstr]) + ".h5"
        if with_defects:
            pred_name =  pred_name[:-3] + "_with_defects.h5"
        if len(pred_name) >= 256:
            pred_name = str(hash(pred_name[:-3])) + ".h5"
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        if not os.path.exists(pred_folder):
            os.mkdir(pred_folder)
        pred_path = os.path.join(pred_folder, pred_name)

        # see if the rf is already learned and predicted, otherwise learn it
        if os.path.exists(pred_path):
            print "Loading prediction from:"
            print pred_path
            return vigra.readHDF5(pred_path, 'data')

    # get the random forest(s)
    rfs = learn_rf(cache_folder,
        trainsets,
        seg_id_train,
        feature_aggregator,
        exp_params,
        trainstr,
        paramstr,
        with_defects,
        use_2rfs)

    if use_2rfs:
        rf_xy = rfs[0]
        rf_z  = rfs[1]
        assert rf_xy.treeCount() == rf_z.treeCount()
        if len(rfs) == 3:
            assert with_defects
            rf_defects = rfs[2]
            assert rf_defects.treeCount() == rf_xy.treeCount()
        else:
            assert len(rfs) == 2
    else:
        rf = rfs[0]
        if len(rfs) == 2:
            assert with_defects
            rf_defects = rfs[1]
            assert rf_defects.treeCount() == rf.treeCount()
        else:
            assert len(rfs) == 1

    # get the training features
    features_test  = feature_aggregator( ds_test, seg_id_test )

    if with_defects and ds_test.has_defects:
        skip_transition = features_test.shape[0] - get_skip_edges(
                ds_test,
                seg_id_test).shape[0]
        features_test_skip = features_test[skip_transition:]
        features_test = features_test[:skip_transition]

    # predict
    if use_2rfs:
        edge_indications = modified_edge_indications(ds_test, seg_id_test)[skip_transition:] \
                if (with_defects and ds_test.has_defects) else ds_test.edge_indications(seg_id_test)
        pmem_xy = rf_xy.predictProbabilities(features_test[edge_indications==1].astype('float32'),
                n_threads = exp_params.n_threads)[:,1]
        pmem_z  = rf_z.predictProbabilities(features_test[edge_indications==0].astype('float32'),
                n_threads = exp_params.n_threads)[:,1]
        pmem_test = np.zeros_like(edge_indications, dtype = 'float32')
        pmem_test[edge_indications==1] = pmem_xy
        pmem_test[edge_indications==0] = pmem_z
        n_trees = rf_xy.treeCount()
    else:
        print "Start predicting random forest"
        pmem_test = rf.predictProbabilities( features_test.astype('float32'),
            n_threads = exp_params.n_threads)[:,1]
        n_trees = rf.treeCount()

    if with_defects and ds_test.has_defects:
        print "Start predicting defect random forest"
        pmem_skip = rf_defects.predictProbabilities( features_test_skip.astype('float32'),
            n_threads = exp_params.n_threads)[:,1]
        pmem_test = np.concatenate([pmem_test, pmem_skip])

    # normalize by the number of trees and remove nans
    pmem_test /= n_trees
    pmem_test[np.isnan(pmem_test)] = .5
    pmem_test[np.isinf(pmem_test)] = .5
    assert pmem_test.max() <= 1.

    if cache_folder is not None:
        vigra.writeHDF5(pmem_test, pred_path, 'data')

    return pmem_test
