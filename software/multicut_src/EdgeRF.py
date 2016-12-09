import numpy as np
import vigra
import cPickle as pickle
import os

from DataSet import DataSet
from ExperimentSettings import ExperimentSettings
from Tools import edges_to_volume

from sklearn.ensemble import RandomForestClassifier



# set cache folder to None if you dont want to cache the resulting rf
def learn_and_predict_rf_from_gt(cache_folder,
        trainsets, ds_test,
        seg_id_train, seg_id_test,
        feature_list, exp_params):

    # this should also work for cutouts, because they inherit from dataset
    assert isinstance(trainsets, DataSet) or isinstance(trainsets, list)
    assert isinstance(ds_test, DataSet)
    assert isinstance(exp_params, ExperimentSettings)

    # if we don't have a list, put the dataset in, so we don't need to
    # make two diverging code strangs....
    if isinstance(trainsets, DataSet):
        trainsets = [ trainsets, ]

    features_train = []
    labels_train   = []
    for cutout in trainsets:

        assert isinstance(cutout, DataSet)
        assert cutout.has_gt

        features_cut = cutout.local_feature_aggregator(seg_id_train, feature_list,
                exp_params.anisotropy_factor, exp_params.use_2d)

        if exp_params.learn_fuzzy:
            labels_cut = cutout.edge_gt_fuzzy(seg_id_train,
                    exp_params.positive_threshold, exp_params.negative_threshold)
        else:
            labels_cut = cutout.edge_gt(seg_id_train)

        assert labels_cut.shape[0] == features_cut.shape[0]

        # we set all labels that are going to be ignored to 0.5

        # set ignore mask to 0.5
        if exp_params.use_ignore_mask:
            ignore_mask = cutout.ignore_mask(seg_id_train)
            assert ignore_mask.shape[0] == labels_cut.shape[0]
            labels_cut[ ignore_mask ] = 0.5

        # set z edges to 0.5
        if exp_params.learn_2d:
            edge_indications = cutout.edge_indications(seg_id_train)
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
        labels_cut   = labels_cut[labeled]

        features_train.append(features_cut)
        labels_train.append(labels_cut)

    features_train = np.concatenate(features_train)
    labels_train = np.concatenate(labels_train)

    assert features_train.shape[0] == labels_train.shape[0]
    print np.unique(labels_train)
    assert all( np.unique(labels_train) == np.array([0, 1]) ), "Unique labels: " + str(np.unique(labels_train))

    features_test  = ds_test.local_feature_aggregator(seg_id_test, feature_list,
            exp_params.anisotropy_factor, exp_params.use_2d)
    assert features_train.shape[1] == features_test.shape[1]

    # strings for caching

    # str for trainingsets
    trainstr = "_".join([ds.ds_name for ds in trainsets ]) + "_" + str(seg_id_train)
    # str for testset
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    # str for all relevant params
    # TODO once we don't need caches any longer: str(arg) for arg in feature_list
    paramstr = "_".join( [str(feature_list), str(exp_params.anisotropy_factor),
        str(exp_params.learn_2d), str(exp_params.learn_fuzzy),
        str(exp_params.n_trees), str(exp_params.negative_threshold),
        str(exp_params.positive_threshold), str(exp_params.use_2d),
        str(exp_params.use_ignore_mask)] )

    rf_verbosity = 0
    if exp_params.verbose:
        rf_verbosity = 2

    # Only cache if we have a valid caching folder
    if cache_folder is not None:

        pred_folder = os.path.join(cache_folder, "pred_" + trainstr)
        pred_name = "prediction_" + "_".join([trainstr, teststr, paramstr])

        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        if not os.path.exists(pred_folder):
            os.mkdir(pred_folder)
        pred_path = os.path.join(pred_folder, pred_name)

        # see if the rf is already learned and predicted, otherwise learn it
        if not os.path.exists(pred_path):
            rf = learn_rf(cache_folder, trainstr, paramstr,
                    seg_id_train, features_train,
                    labels_train, exp_params.n_trees, exp_params.n_threads, rf_verbosity)
            # we only keep the second channel, because this corresponds to the probability for being a real membrane
            pmem_test = rf.predict_proba( features_test )[:,1]
            vigra.writeHDF5(pmem_test, pred_path, "data")
        else:
            pmem_test = vigra.readHDF5(pred_path, "data")

    else:
        rf = learn_rf(cache_folder, trainstr, paramstr,
                seg_id_train, features_train,
                labels_train, exp_params.n_trees, exp_params.n_threads, rf_verbosity)
        # we only keep the second channel, because this corresponds to the probability for being a real membrane
        pmem_test = rf.predict_proba( features_test )[:,1]

    return pmem_test


# set cache folder to None if you dont want to cache the result
def learn_and_predict_anisotropic_rf(cache_folder,
        trainsets, ds_test,
        seg_id_train, seg_id_test,
        feature_list_xy, feature_list_z, exp_params):

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

    for cut in trainsets:

        assert cut.has_gt
        features_xy = cut.local_feature_aggregator(seg_id_train, feature_list_xy,
                exp_params.anisotropy_factor, False)

        features_z = cut.local_feature_aggregator(seg_id_train, feature_list_z,
                exp_params.anisotropy_factor, exp_params.use_2d)

        if exp_params.learn_fuzzy:
            labels = cut.edge_gt_fuzzy(seg_id_train,
                    exp_params.positive_threshold, exp_params.negative_threshold)
        else:
            labels = cut.edge_gt(seg_id_train)

        # we set all labels that are going to be ignored to 0.5

        # set ignore mask to 0.5
        if exp_params.use_ignore_mask:
            ignore_mask = cut.ignore_mask(seg_id_train)
            assert ignore_mask.shape[0] == labels.shape[0]
            labels[ np.logical_not(ignore_mask) ] = 0.5

        labeled = labels != 0.5

        edge_indications = cut.edge_indications(seg_id_train)[labeled]
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

    edge_indications_test = ds_test.edge_indications(seg_id_test)

    features_test_xy  = ds_test.local_feature_aggregator(seg_id_test, feature_list_xy,
            exp_params.anisotropy_factor, False)[edge_indications_test == 1]

    features_test_z  = ds_test.local_feature_aggregator(seg_id_test, feature_list_z,
            exp_params.anisotropy_factor, exp_params.use_2d)[edge_indications_test == 0]


    pmem_xy = rf_xy.predict_proba( features_test_xy )[:,1]
    pmem_z  = rf_z.predict_proba( features_test_z )[:,1]

    pmem_test = np.zeros_like( edge_indications_test)
    pmem_test[edge_indications_test == 1] = pmem_xy
    pmem_test[edge_indications_test == 0] = pmem_z

    return pmem_test


def learn_rf(cache_folder, trainstr, paramstr, seg_id,
        features, labels, n_trees = 500, n_threads = 1, verbose = 0, oob = False):

    assert labels.shape[0] == features.shape[0], str(labels.shape[0]) + " , " + str(features.shape[0])

    # cache
    if cache_folder is not None:

        rf_folder = os.path.join(cache_folder, "rf_" + trainstr)
        rf_name = "rf_" + "_".join( [trainstr, paramstr] )

        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)

        rf_path   = os.path.join(rf_folder, rf_name)

        if not os.path.exists(rf_path):
            rf = RandomForestClassifier(n_estimators = n_trees, n_jobs = n_threads,
                    oob_score = oob, verbose = verbose)
            rf.fit( features, labels.astype(np.uint32).ravel() )
            if oob:
                oob_err = 1. - rf.oob_score_
                print "Random Forest was trained with OOB Error:", oob_err
            with open(rf_path, 'w') as f:
                pickle.dump(rf, f)
        else:
            with open(rf_path, 'r') as f:
                rf = pickle.load(f)
    # no caching
    else:
        rf = RandomForestClassifier(n_estimators = n_trees, n_jobs = n_threads,
                oob_score = True, verbose = verbose)
        rf.fit( features, labels.astype(np.uint32).ravel() )
        oob = 1. - rf.oob_score_

        print "Random Forest was trained with OOB Error:", oob

    return rf
