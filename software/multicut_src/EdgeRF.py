import numpy as np
import vigra
import os
from functools import partial

from DataSet import DataSet
from defect_handling import modified_edge_features, modified_region_features, modified_topology_features, modified_edge_indications, modified_edge_gt
from ExperimentSettings import ExperimentSettings
from Tools import edges_to_volume

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
        features.append(ds.edge_features(seg_id, 0, anisotropy_factor ))
    if "prob" in feature_list:
        features.append(ds.edge_features(seg_id, 1, anisotropy_factor ))
    if "affinities" in feature_list:
        features.append(ds.edge_features_from_affinity_maps(seg_id, 1, anisotropy_factor ))
    if "extra_input" in feature_list:
        features.append(ds.edge_features(seg_id, 2, anisotropy_factor ))
    if "reg" in feature_list:
        features.append(ds.region_features(seg_id, 0,
            ds._adjacent_segments(seg_id), False ) )
    if "topo" in feature_list:
        features.append(ds.topology_features(seg_id, use_2d ))
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
        n_bins,
        bin_threshold,
        anisotropy_factor = 1.,
        use_2d = False):

    assert seg_id < ds.n_seg, str(seg_id) + " , " + str(ds.n_seg)
    assert anisotropy_factor >= 1., "Finer resolution in z-direction is not supported"
    for feat in feature_list:
        assert feat in ("raw", "prob", "affinities", "extra_input", "reg", "topo"), feat
    features = []
    if "raw" in feature_list:
        features.append(modified_edge_features(ds, seg_id, 0, anisotropy_factor, n_bins, bin_threshold ))
    if "prob" in feature_list:
        features.append(modified_edge_features(ds, seg_id, 1, anisotropy_factor, n_bins, bin_threshold ))
    if "affinities" in feature_list:
        features.append(modified_edge_features_from_affinity_maps(ds, seg_id, 1, anisotropy_factor, n_bins, bin_threshold ))
    if "extra_input" in feature_list:
        features.append(modified_edge_features(ds, seg_id, 2, anisotropy_factor, n_bins, bin_threshold ))
    if "reg" in feature_list:
        features.append(modified_region_features(ds, seg_id, 0, ds._adjacent_segments(seg_id), False, n_bins, bin_threshold ) )
    if "topo" in feature_list:
        features.append(modified_topology_features(ds, seg_id, use_2d, n_bins, bin_threshold ))
    #if "curve" in feature_list:
    #    features.append(ds.curvature_features(seg_id))

    return np.concatenate(features, axis = 1)


# set cache folder to None if you dont want to cache the resulting rf
def learn_and_predict_rf_from_gt(cache_folder,
        trainsets, ds_test,
        seg_id_train, seg_id_test,
        feature_list, exp_params,
        with_defects = False,
        n_bins = 0,
        bin_threshold = 0):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)
    assert isinstance(ds_test, DataSet)
    assert isinstance(exp_params, ExperimentSettings)

    # if we don't have a list, put the dataset in, so we don't need to
    # make two diverging code strangs....
    if isinstance(trainsets, DataSet):
        trainsets = [ trainsets, ]

    if with_defects:
        assert n_bins > 0
        assert bin_threshold > 0
        feature_aggregator = partial( local_feature_aggregator_with_defects,
            feature_list = feature_list, n_bins = n_bins,
            bin_threshold = bin_threshold, anisotropy_factor = exp_params.anisotropy_factor,
            use_2d = exp_params.use_2d )
    else:
        feature_aggregator = partial( local_feature_aggregator,
            feature_list = feature_list, anisotropy_factor = exp_params.anisotropy_factor,
            use_2d = exp_params.use_2d )

    features_train = []
    labels_train   = []
    for cutout in trainsets:

        assert isinstance(cutout, DataSet)
        assert cutout.has_gt

        features_cut = feature_aggregator( cutout, seg_id_train )

        if exp_params.learn_fuzzy:
            if with_defects:
                raise AttributeError("Fuzzy learning not supported ford defect pipeline yet")
            labels_cut = cutout.edge_gt_fuzzy(seg_id_train,
                    exp_params.positive_threshold, exp_params.negative_threshold)
        else:
            labels_cut = modified_edge_gt(cutout, seg_id_train, n_bins, bin_threshold) if with_defects else cutout.edge_gt(seg_id_train)

        assert labels_cut.shape[0] == features_cut.shape[0]

        # we set all labels that are going to be ignored to 0.5

        # set ignore mask to 0.5
        if exp_params.use_ignore_mask and not with_defects: # ignore mask not yet supported for defect pipeline
            ignore_mask = cutout.ignore_mask(seg_id_train)
            assert ignore_mask.shape[0] == labels_cut.shape[0]
            labels_cut[ ignore_mask ] = 0.5

        # set z edges to 0.5
        if exp_params.learn_2d:
            edge_indications = modified_edge_indications(cutout, seg_id_train) if with_defects else cutout.edge_indications(seg_id_train)
            labels_cut[edge_indications == 0] = 0.5

        # inspect edges for debugging
        if False:
            labels_for_vol = np.zeros(labels_cut.shape, dtype = np.uint32)
            labels_for_vol[labels_cut == 1.]  = 1
            labels_for_vol[labels_cut == 0.]  = 2
            labels_for_vol[labels_cut == 0.5] = 5
            # set z-labels to zero
            #labels_for_vol[cutout.edge_indications(seg_id_train) == 0] = 0

            edge_vol = edges_to_volume(cutout._rag(seg_id_train), labels_for_vol)

            from volumina_viewer import volumina_n_layer
            volumina_n_layer([cutout.inp(0),
                cutout.seg(seg_id_train).astype(np.uint32),
                cutout.gt().astype(np.uint32),
                edge_vol.astype(np.uint32)])

        labeled = labels_cut != 0.5

        features_cut = features_cut[labeled]
        labels_cut   = labels_cut[labeled].astype('uint8')

        features_train.append(features_cut)
        labels_train.append(labels_cut)

    features_train = np.concatenate(features_train)
    labels_train = np.concatenate(labels_train)

    assert features_train.shape[0] == labels_train.shape[0]
    assert all( np.unique(labels_train) == np.array([0, 1]) ), "Unique labels: " + str(np.unique(labels_train))

    features_test  = feature_aggregator( ds_test, seg_id_test )
    assert features_train.shape[1] == features_test.shape[1]

    # strings for caching

    # str for trainingsets
    trainstr = "_".join([ds.ds_name for ds in trainsets ]) + "_" + str(seg_id_train)
    # str for testset
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    # str for all relevant params
    # TODO once we don't need caches any longer: str(arg) for arg in feature_list
    paramstr = "_".join( ["_".join(feature_list), str(exp_params.anisotropy_factor),
        str(exp_params.learn_2d), str(exp_params.learn_fuzzy),
        str(exp_params.n_trees), str(exp_params.negative_threshold),
        str(exp_params.positive_threshold), str(exp_params.use_2d),
        str(exp_params.use_ignore_mask)] )

    # Only cache if we have a valid caching folder
    if cache_folder is not None:

        pred_folder = os.path.join(cache_folder, "pred_" + trainstr)
        pred_name = "prediction_" + "_".join([trainstr, teststr, paramstr]) + ".h5"

        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        if not os.path.exists(pred_folder):
            os.mkdir(pred_folder)
        pred_path = os.path.join(pred_folder, pred_name)

        # see if the rf is already learned and predicted, otherwise learn it
        if not os.path.exists(pred_path):
            rf = learn_rf(cache_folder, trainstr, paramstr,
                    seg_id_train, features_train,
                    labels_train, exp_params.n_trees, exp_params.n_threads)
            # we only keep the second channel, because this corresponds to the probability for being a real membrane
            pmem_test = rf.predictProbabilities( features_test.astype('float32'),
                n_threads = exp_params.n_threads)[:,1]
            pmem_test /= rf.treeCount()
            vigra.writeHDF5(pmem_test, pred_path, "data")
            # FIXME sometimes there are some nans -> just replace them for now, but this should be fixed
            pmem_test[np.isnan(pmem_test)] = .5
            #assert not np.isnan(pmem_test).any(), str(np.isnan(pmem_test).sum())
            #if np.isnan(pmem_test).any():
            #    import ipdb
            #    ipdb.set_trace()
        else:
            pmem_test = vigra.readHDF5(pred_path, "data")

    else:
        rf = learn_rf(cache_folder, trainstr, paramstr,
                seg_id_train, features_train,
                labels_train, exp_params.n_trees, exp_params.n_threads)
        # we only keep the second channel, because this corresponds to the probability for being a real membrane
        pmem_test = rf.predictProbabilities( features_test.astype('float32'),
            n_threads = exp_params.n_threads)[:,1]

    return pmem_test


# TODO implement caching
# set cache folder to None if you dont want to cache the result
def learn_and_predict_anisotropic_rf(cache_folder,
        trainsets, ds_test,
        seg_id_train, seg_id_test,
        feature_list_xy, feature_list_z,
        exp_params,
        with_defects = False,
        n_bins = 0,
        bin_threshold = 0):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)
    assert isinstance(ds_test, DataSet)
    assert isinstance(exp_params, ExperimentSettings)

    # if we don't have a list, put the dataset in, so we don't need to
    # make two diverging code strangs....
    if isinstance(trainsets, DataSet):
        trainsets = [ trainsets, ]

    features_train_xy = []
    features_train_z = []

    labels_train_xy = []
    labels_train_z  = []

    if with_defects:
        assert n_bins > 0
        assert bin_threshold > 0
        feature_aggregator = partial( local_feature_aggregator_with_defects,
            n_bins = n_bins, bin_threshold = bin_threshold,
            anisotropy_factor = exp_params.anisotropy_factor)
    else:
        feature_aggregator = partial( local_feature_aggregator,
            anisotropy_factor = exp_params.anisotropy_factor)

    for cut in trainsets:

        assert cut.has_gt
        features_xy = feature_aggregator(cut, seg_id_train, feature_list = feature_list_xy, use_2d = False)
        features_z = feature_aggregator(cut, seg_id_train, feature_list = feature_list_z, use_2d = exp_params.use_2d)

        if exp_params.learn_fuzzy:
            if with_defects:
                raise AttributeError("Fuzzy learning not supported ford defect pipeline yet")
            labels_cut = cut.edge_gt_fuzzy(seg_id_train,
                    exp_params.positive_threshold, exp_params.negative_threshold)
        else:
            labels_cut = modified_edge_gt(cut, seg_id_train, n_bins, bin_threshold) if with_defects else cut.edge_gt(seg_id_train)

        # we set all labels that are going to be ignored to 0.5

        # set ignore mask to 0.5
        if exp_params.use_ignore_mask and not with_defects: # ignore mask not yet supported for defects
            ignore_mask = cut.ignore_mask(seg_id_train)
            assert ignore_mask.shape[0] == labels.shape[0]
            labels[ np.logical_not(ignore_mask) ] = 0.5

        labeled = labels != 0.5

        edge_indications = modified_edge_indications(cut, seg_id_train)[labeled] if with_defects else cut.edge_indications(seg_id_train)[labeled]
        features_xy = features_xy[labeled][edge_indications==1]
        features_z = features_z[labeled][edge_indications==0]
        labels   = labels[labeled]

        features_train_xy.append(features_xy)
        features_train_z.append(features_z)

        labels_train_xy.append(labels[edge_indications==1])
        labels_train_z.append(labels[edge_indications==0])

    features_train_xy = np.concatenate(features_train_xy)
    labels_train_xy = np.concatenate(labels_train_xy)

    features_train_z = np.concatenate(features_train_z)
    labels_train_z = np.concatenate(labels_train_z)

    # no stupid caching for now!
    rf_xy = RandomForestClassifier(n_estimators = exp_params.n_trees, n_jobs = -1)
    rf_xy.fit( features_train_xy, labels_train_xy.astype(np.uint32) )#.ravel() )

    rf_z = RandomForestClassifier(n_estimators = exp_params.n_trees, n_jobs = -1)
    rf_z.fit( features_train_z, labels_train_z.astype(np.uint32) )#.ravel() )

    edge_indications_test = modified_edge_indications(ds_test, seg_id_test) if with_defects else ds_test.edge_indications(seg_id_test)

    features_test_xy  = feature_aggregator(ds_test, seg_id_test,
            feature_list = feature_list_xy, use_2d = False)[edge_indications_test == 1]

    features_test_z  = feature_aggregator(ds_test, seg_id_test,
            feature_list = feature_list_z, use_2d = exp_params.use_2d)[edge_indications_test == 0]

    pmem_xy = rf_xy.predictProbabilities( features_test_xy.astype('float32'),
            n_threads = exp_params.n_threads)[:,1]
    pmem_z  = rf_z.predictProbabilities( features_test_z.astype('float32'),
            n_threads = exp_params.n_threads)[:,1]

    pmem_test = np.zeros_like( edge_indications_test)
    pmem_test[edge_indications_test == 1] = pmem_xy
    pmem_test[edge_indications_test == 0] = pmem_z
    pmem_test /= rf_xy.treeCount()
    # FIXME sometimes there are some nans -> just replace them for now, but this should be fixed
    pmem_test[np.isnan(pmem_test)] = .5

    return pmem_test


def learn_rf(cache_folder, trainstr, paramstr, seg_id,
        features, labels, n_trees = 500, n_threads = 1, verbose = 0, oob = False):

    assert labels.shape[0] == features.shape[0], str(labels.shape[0]) + " , " + str(features.shape[0])

    # cache
    if cache_folder is not None:
        rf_folder = os.path.join(cache_folder, "rf_" + trainstr)
        rf_name = "rf_" + "_".join( [trainstr, paramstr] ) + ".h5"
        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)
        rf_path   = os.path.join(rf_folder, rf_name)
        if not os.path.exists(rf_path):
            rf = RandomForest(features.astype('float32'), labels.astype('uint32').ravel(),
                treeCount = n_trees,
                n_threads = n_threads)
            # TODO expose OOB in vigra
            #if oob:
            #    oob_err = 1. - rf.oob_score_
            #    print "Random Forest was trained with OOB Error:", oob_err
            rf.writeHDF5(rf_path, 'data')
        else:
            rf = RandomForest(rf_path, 'data')
    # no caching
    else:
        rf = RandomForest(features.astype('float32'), labels.astype('uint32').ravel(),
            treeCount = n_trees,
            n_threads = n_threads)
        # TODO oob in vigra
        #oob = 1. - rf.oob_score_
        #print "Random Forest was trained with OOB Error:", oob
    return rf
