import numpy as np
import vigra
import h5py
import os
from functools import partial

from defect_handling import modified_edge_features, modified_region_features, modified_topology_features
from defect_handling import modified_edge_features_from_affinity_maps
from defect_handling import modified_edge_indications, modified_edge_gt, modified_edge_gt_fuzzy
from defect_handling import get_skip_edges, modified_adjacency, get_ignore_edge_ids
from ExperimentSettings import ExperimentSettings

try:
    from sklearn.ensemble import RandomForestClassifier as RFType
    use_sklearn = True
    import cPickle as pickle
    print "Using sklearn random forest"
except ImportError:
    RFType = vigra.learning.RandomForest3
    use_sklearn = False
    print "Using vigra random forest 3"


# wrapper for sklearn / random forest
class RandomForest(object):

    def __init__(
            self,
            train_data,
            train_labels,
            n_trees,
            n_threads,
            max_depth=None
    ):

        if isinstance(train_data, str) and train_data == '__will_deserialize__':
            return
        else:
            assert isinstance(train_data, np.ndarray)

        assert train_data.shape[0] == train_labels.shape[0]
        self.n_threads = n_threads
        self.n_trees = n_trees
        self.max_depth = max_depth

        if use_sklearn:
            self._learn_rf_sklearn(train_data, train_labels)
        else:
            self._learn_rf_vigra(train_data, train_labels)

    @classmethod
    def load_from_file(self, file_path, key, n_threads):
        self = self('__will_deserialize__', None, None, n_threads)
        if use_sklearn:
            # remove '.h5' from the file path and add the key
            save_path = os.path.join(file_path, "%s.pkl" % key)
            with open(save_path) as f:
                rf = pickle.load(f)
            self.n_trees = rf.n_estimators
        else:
            save_path = file_path + ".h5"
            rf = RFType(save_path, key)
            self.n_trees = rf.treeCount()
        self.rf = rf
        return self

    @staticmethod
    def has_defect_rf(file_path):
        if use_sklearn:
            return os.path.exists(os.path.join(file_path, "rf_defects.pkl"))
        else:
            with h5py.File(file_path + '.h5') as f:
                return 'rf_defects' in f.keys()

    @classmethod
    def is_cached(self, file_path):
        save_path = file_path if use_sklearn else file_path + ".h5"
        return os.path.exists(save_path)

    def _learn_rf_sklearn(self, train_data, train_labels):
        self.rf = RFType(
            n_estimators=self.n_trees,
            n_jobs=self.n_threads,
            verbose=2 if ExperimentSettings().verbose else 0,
            max_depth=self.max_depth
        )
        self.rf.fit(train_data, train_labels)

    def _learn_rf_vigra(self, train_data, train_labels):
        self.rf = RFType(
            train_data,
            train_labels,
            treeCount=self.n_trees,
            n_threads=self.n_threads,
            max_depth=self.max_depth if self.max_depth is not None else 0
        )

    def predict_probabilities(self, test_data):
        if use_sklearn:
            return self._predict_sklearn(test_data)
        else:
            return self._predict_vigra(test_data)

    def _predict_sklearn(self, test_data):
        return self.rf.predict_proba(test_data)

    def _predict_vigra(self, test_data):
        prediction = self.rf.predictProbabilities(test_data, n_threads=self.n_threads)
        # normalize the prediction
        prediction /= self.n_trees
        # normalize by the number of trees and remove nans
        prediction[np.isnan(prediction)] = .5
        prediction[np.isinf(prediction)] = .5
        assert prediction.max() <= 1.
        return prediction

    def write(self, file_path, key):
        if use_sklearn:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            save_path = os.path.join(file_path, "%s.pkl" % (key))
            with open(save_path, 'w') as f:
                pickle.dump(self.rf, f)
        else:
            save_path = file_path + ".h5"
            self.rf.writeHDF5(file_path, key)


# toplevel convenience function for features
# aggregates all the features given in feature list:
# possible valus: "raw" -> edge features from raw_data
# "prob" -> edge features from probability maps
# "reg"  -> features from region statistics
# "topo" -> topological features
def local_feature_aggregator(
        ds,
        seg_id,
        feature_list,
        anisotropy_factor=1.,
        use_2d=False
):

    assert seg_id < ds.n_seg, str(seg_id) + " , " + str(ds.n_seg)
    assert anisotropy_factor >= 1., "Finer resolution in z-direction is not supported"
    for feat in feature_list:
        assert feat in ("raw", "prob", "affinities", "extra_input", "reg", "topo"), feat
    features = []
    if "raw" in feature_list:
        features.append(ds.edge_features(seg_id, 0, anisotropy_factor))
    if "prob" in feature_list:
        features.append(ds.edge_features(seg_id, 1, anisotropy_factor))
    if "affinities" in feature_list:
        features.append(
            ds.edge_features_from_affinity_maps(
                seg_id, (1, 2), anisotropy_factor, ExperimentSettings().affinity_z_direction)
        )
    if "extra_input" in feature_list:
        features.append(ds.edge_features(seg_id, 2, anisotropy_factor))
    if "reg" in feature_list:
        features.append(
            ds.region_features(seg_id, 0, ds.uv_ids(seg_id), False)
        )
    if "topo" in feature_list:
        features.append(ds.topology_features(seg_id, use_2d))

    return np.concatenate(features, axis=1)


# toplevel convenience function for features
# aggregates all the features given in feature list:
# possible valus: "raw" -> edge features from raw_data
# "prob" -> edge features from probability maps
# "reg"  -> features from region statistics
# "topo" -> topological features
def local_feature_aggregator_with_defects(
        ds,
        seg_id,
        feature_list,
        anisotropy_factor=1.,
        use_2d=False
):

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
        features.append(modified_edge_features(ds, seg_id, 1, anisotropy_factor))
    if "affinities" in feature_list:
        features.append(modified_edge_features_from_affinity_maps(
            ds,
            seg_id,
            (1, 2),
            anisotropy_factor,
            ExperimentSettings().affinity_z_direction)
        )
    if "extra_input" in feature_list:
        features.append(modified_edge_features(ds, seg_id, 2, anisotropy_factor))
    if "reg" in feature_list:
        features.append(modified_region_features(ds, seg_id, 0, ds.uv_ids(seg_id), False))
    if "topo" in feature_list:
        features.append(modified_topology_features(ds, seg_id, use_2d))

    return np.concatenate(features, axis=1)


# edge masking:
# we set all labels that are going to be ignored to 0.5
def mask_edges(
        ds,
        seg_id,
        labels,
        uv_ids,
        with_defects
):

    labeled = np.ones_like(labels, dtype=bool)
    # set ignore mask to 0.5
    if ExperimentSettings().use_ignore_mask:  # ignore mask not yet supported for defect pipeline
        ignore_mask = ds.ignore_mask(seg_id, uv_ids, with_defects)
        assert ignore_mask.shape[0] == labels.shape[0]
        labeled[ignore_mask] = False

    # ignore all edges that are connected to the ignore segment label in the seg mask
    if ds.has_seg_mask:
        ignore_mask = (uv_ids == ExperimentSettings().ignore_seg_value).any(axis=1)
        assert ignore_mask.shape[0] == labels.shape[0]
        labeled[ignore_mask] = False

    # set z edges to 0.5
    if ExperimentSettings().learn_2d:
        edge_indications = modified_edge_indications(ds, seg_id) if with_defects else ds.edge_indications(seg_id)
        labeled[edge_indications == 0] = False

    # mask the ignore edges
    if with_defects and ds.has_defects:
        ignore_edge_ids = get_ignore_edge_ids(ds, seg_id)
        if ignore_edge_ids.size:
            labeled[ignore_edge_ids] = False

    return labeled


# FIXME: For now, we don't support different feature strings for the edge types
# if features should differ for the edge types (e.g. affinities), these need to be
# already merged in the feature computation
def _learn_seperate_rfs(
        trainsets,
        seg_id,
        features,
        labels,
        labeled,
        rf_path,
        features_skip=None,
        labels_skip=None,
        with_defects=False
):

    assert len(trainsets) == len(features)
    assert len(labels) == len(features)
    assert len(labels) == len(labeled), "%i, %i" % (len(labels), len(labeled))

    if with_defects:
        skip_transitions = [
            modified_adjacency(ds, seg_id).shape[0] - get_skip_edges(ds, seg_id).shape[0]
            if ds.has_defects else ds.uv_ids(seg_id).shape[0] for i, ds in enumerate(trainsets)
        ]

    all_indications = [
        modified_edge_indications(ds, seg_id)[:skip_transitions[i]][labeled[i]]
        if (with_defects and ds.has_defects) else ds.edge_indications(seg_id)[labeled[i]]
        for i, ds in enumerate(trainsets)
    ]

    features_xy = np.concatenate(
        [features[i][indications == 1] for i, indications in enumerate(all_indications)]
    )
    labels_xy = np.concatenate(
        [labels[i][indications == 1] for i, indications in enumerate(all_indications)]
    )

    features_z = np.concatenate(
        [features[i][indications == 0] for i, indications in enumerate(all_indications)]
    )
    labels_z = np.concatenate(
        [labels[i][indications == 0] for i, indications in enumerate(all_indications)]
    )

    assert features_xy.shape[0] == labels_xy.shape[0]
    assert features_z.shape[0] == labels_z.shape[0]
    assert all(np.unique(labels_xy) == np.array([0, 1])), "unique labels: " + str(np.unique(labels_xy))
    assert all(np.unique(labels_z) == np.array([0, 1])), "unique labels: " + str(np.unique(labels_z))

    print "Start learning random forest for xy edges"
    rf_xy = RandomForest(
        features_xy.astype('float32'),
        labels_xy,
        n_trees=ExperimentSettings().n_trees,
        n_threads=ExperimentSettings().n_threads
    )

    print "Start learning random forest for z edges"
    rf_z = RandomForest(
        features_z.astype('float32'),
        labels_z,
        n_trees=ExperimentSettings().n_trees,
        n_threads=ExperimentSettings().n_threads
    )

    if features_skip is not None:
        assert labels_skip is not None
        print "Start learning defect random forest"
        rf_defects = RandomForest(
            features_skip.astype('float32'),
            labels_skip,
            n_trees=ExperimentSettings().n_trees,
            n_threads=ExperimentSettings().n_threads
        )

    if rf_path is not None:
        rf_xy.write(rf_path, 'rf_xy')
        rf_z.write(rf_path, 'rf_z')
        if features_skip is not None:
            rf_defects.write(rf_path, 'rf_defects')

    if features_skip is not None:
        return [rf_xy, rf_z, rf_defects]
    else:
        return [rf_xy, rf_z]


def _learn_single_rfs(
        features,
        labels,
        rf_path,
        features_skip=None,
        labels_skip=None,
        with_defects=False
):

    features = np.concatenate(features)
    labels  = np.concatenate(labels)

    assert features.shape[0] == labels.shape[0]
    assert all(np.unique(labels) == np.array([0, 1])), "unique labels: " + str(np.unique(labels))

    print "Start learning random forest"
    rf = RandomForest(
        features.astype('float32'),
        labels,
        n_trees=ExperimentSettings().n_trees,
        n_threads=ExperimentSettings().n_threads
    )

    if features_skip is not None:
        assert labels_skip is not None
        print "Start learning defect random forest"
        rf_defects = RandomForest(
            features_skip.astype('float32'),
            labels_skip,
            n_trees=ExperimentSettings().n_trees,
            n_threads=ExperimentSettings().n_threads
        )

    if rf_path is not None:
        rf.write(rf_path, 'rf')
        if with_defects:
            rf_defects.write(rf_path, 'rf_defects')

    if features_skip is not None:
        return [rf, rf_defects]
    else:
        return [rf]


def learn_rf(
        trainsets,
        seg_id,
        feature_aggregator,
        trainstr,
        paramstr,
        with_defects=False,
        use_2rfs=False
):

    cache_folder = ExperimentSettings().rf_cache_folder
    if cache_folder is not None:  # we use caching for the rf => look if already exists
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        rf_folder = os.path.join(cache_folder, "rf_" + trainstr)
        rf_name = "rf_" + "_".join([trainstr, paramstr])
        if len(rf_name) > 255:
            rf_name = str(hash(rf_name))

        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)
        rf_path   = os.path.join(rf_folder, rf_name)
        if RandomForest.is_cached(rf_path):
            print "Loading random forest from:"
            print rf_path
            if use_2rfs:
                rfs = [
                    RandomForest.load_from_file(rf_path, 'rf_xy', ExperimentSettings().n_threads),
                    RandomForest.load_from_file(rf_path, 'rf_z', ExperimentSettings().n_threads)
                ]
            else:
                rfs = [RandomForest.load_from_file(rf_path, 'rf', ExperimentSettings().n_threads)]
            # we need to check if the defect rf actually exists
            if RandomForest.has_defect_rf(rf_path):
                assert with_defects
                rfs.append(RandomForest.load_from_file(rf_path, 'rf_defects', ExperimentSettings().n_threads))
            return rfs

    features_train = []
    labels_train   = []
    labeled_train  = []

    if with_defects:
        features_skip = []
        labels_skip   = []

    # iterate over the datasets in our traninsets
    for ds in trainsets:

        assert ds.has_gt

        features_sub = feature_aggregator(ds, seg_id)
        uv_ids = modified_adjacency(ds, seg_id) if with_defects and ds.has_defects \
            else ds.uv_ids(seg_id)
        assert features_sub.shape[0] == uv_ids.shape[0]

        if ExperimentSettings().learn_fuzzy:
            labels_sub = modified_edge_gt_fuzzy(
                ds,
                seg_id,
                ExperimentSettings().positive_threshold,
                ExperimentSettings().negative_threshold
            ) if with_defects and ds.has_defects else \
                ds.edge_gt_fuzzy(
                    seg_id,
                    ExperimentSettings().positive_threshold,
                    ExperimentSettings().negative_threshold)
        else:
            labels_sub = modified_edge_gt(
                ds,
                seg_id) if with_defects and ds.has_defects else ds.edge_gt(seg_id)

        assert labels_sub.shape[0] == features_sub.shape[0], "%i, %i" % (labels_sub.shape[0], features_sub.shape[0])

        # first, find the data points that are actually labeled
        labeled = mask_edges(
            ds,
            seg_id,
            labels_sub,
            uv_ids,
            with_defects
        )

        # inspect the edges FIXME this has dependencies outside of conda, so we can't expose it for now
        if False:
            ds.view_edge_labels(
                seg_id,
                with_defects and ds.has_defects
            )

        # next, if we have defects, seperate the features for the defect random forest from the others
        if with_defects and ds.has_defects:

            # find the transition edge between normal and skip edges
            skip_transition = features_sub.shape[0] - get_skip_edges(ds, seg_id).shape[0]

            # get labeled points for normal and skip edges
            labeled_skip = labeled[skip_transition:]
            labeled = labeled[:skip_transition]

            # get the features and labels for skip edges
            # exclude the non-labeled points directly
            features_skip.append(features_sub[skip_transition:][labeled_skip])
            labels_skip.append(labels_sub[skip_transition:][labeled_skip])

            # get the features and labels for normal edges
            features_sub = features_sub[:skip_transition]
            labels_sub = labels_sub[:skip_transition]

        # exclude the non-labeled points
        features_sub = features_sub[labeled]
        labels_sub   = labels_sub[labeled].astype('uint32')

        features_train.append(features_sub)
        labels_train.append(labels_sub)
        labeled_train.append(labeled)

    if with_defects:
        if features_skip:  # check if any features / labels for skip edges were added
            assert labels_skip
            features_skip = np.concatenate(features_skip)
            labels_skip = np.concatenate(labels_skip)
        else:  # if not set with_defects to false, beacause we can't learn a defect rf
            with_defects = False

    if use_2rfs:
        return _learn_seperate_rfs(
            trainsets, seg_id,
            features_train, labels_train,
            labeled_train,
            rf_path if cache_folder is not None else None,
            features_skip if with_defects else None,
            labels_skip if with_defects else None,
            with_defects
        )
    else:
        return _learn_single_rfs(
            features_train, labels_train,
            rf_path if cache_folder is not None else None,
            features_skip if with_defects else None,
            labels_skip if with_defects else None,
            with_defects
        )


# set cache folder to None if you dont want to cache the resulting rf
def learn_and_predict_rf_from_gt(
        trainsets,
        ds_test,
        seg_id_train,
        seg_id_test,
        feature_list,
        with_defects=False,
        use_2rfs=False
):

    # for only a single ds, put it in a list
    if not isinstance(trainsets, list):
        trainsets = [trainsets]

    if with_defects:
        feature_aggregator = partial(
            local_feature_aggregator_with_defects,
            feature_list=feature_list,
            anisotropy_factor=ExperimentSettings().anisotropy_factor,
            use_2d=ExperimentSettings().use_2d
        )
    else:
        feature_aggregator = partial(
            local_feature_aggregator,
            feature_list=feature_list,
            anisotropy_factor=ExperimentSettings().anisotropy_factor,
            use_2d=ExperimentSettings().use_2d
        )

    # strings for caching
    # str for all relevant params
    paramstr = "_".join(
        ["_".join(feature_list), str(ExperimentSettings().anisotropy_factor),
         str(ExperimentSettings().learn_2d), str(ExperimentSettings().learn_fuzzy),
         str(ExperimentSettings().n_trees), str(ExperimentSettings().negative_threshold),
         str(ExperimentSettings().positive_threshold), str(ExperimentSettings().use_2d),
         str(ExperimentSettings().use_ignore_mask), str(with_defects), str(use_2rfs)]
    )
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    trainstr = "_".join([ds.ds_name for ds in trainsets]) + "_" + str(seg_id_train)

    # we cache this in the ds_test cache folder
    # if caching random forests is activated (== rf_cache_folder is not None)
    if ExperimentSettings().rf_cache_folder is not None:  # cache-folder exists => look if we already have a prediction

        pred_name = "prediction_" + "_".join([trainstr, teststr, paramstr]) + ".h5"
        if len(pred_name) >= 256:
            pred_name = str(hash(pred_name[:-3])) + ".h5"
        pred_path = os.path.join(ds_test.cache_folder, pred_name)

        # see if the rf is already learned and predicted, otherwise learn it
        if os.path.exists(pred_path):
            print "Loading prediction from:"
            print pred_path
            return vigra.readHDF5(pred_path, 'data')

    # get the random forest(s)
    rfs = learn_rf(
        trainsets,
        seg_id_train,
        feature_aggregator,
        trainstr,
        paramstr,
        with_defects,
        use_2rfs)

    if use_2rfs:
        rf_xy = rfs[0]
        rf_z  = rfs[1]
        if len(rfs) == 3:
            assert with_defects
            rf_defects = rfs[2]
        else:
            assert len(rfs) == 2
    else:
        rf = rfs[0]
        if len(rfs) == 2:
            assert with_defects
            rf_defects = rfs[1]
        else:
            assert len(rfs) == 1

    # get the training features
    features_test  = feature_aggregator(ds_test, seg_id_test)

    if with_defects and ds_test.has_defects:
        skip_transition = features_test.shape[0] - get_skip_edges(
            ds_test,
            seg_id_test).shape[0]
        features_test_skip = features_test[skip_transition:]
        features_test = features_test[:skip_transition]

    # predict
    if use_2rfs:
        edge_indications = modified_edge_indications(ds_test, seg_id_test)[:skip_transition] \
            if (with_defects and ds_test.has_defects) else ds_test.edge_indications(seg_id_test)
        pmem_xy = rf_xy.predict_probabilities(features_test[edge_indications == 1].astype('float32'))[:, 1]
        pmem_z  = rf_z.predict_probabilities(features_test[edge_indications == 0].astype('float32'))[:, 1]
        pmem_test = np.zeros_like(edge_indications, dtype='float32')
        pmem_test[edge_indications == 1] = pmem_xy
        pmem_test[edge_indications == 0] = pmem_z
    else:
        print "Start predicting random forest"
        pmem_test = rf.predict_probabilities(features_test.astype('float32'))[:, 1]

    if with_defects and ds_test.has_defects:
        print "Start predicting defect random forest"
        pmem_skip = rf_defects.predict_probabilities(features_test_skip.astype('float32'))[:, 1]
        pmem_test = np.concatenate([pmem_test, pmem_skip])

    if ExperimentSettings().rf_cache_folder is not None:
        vigra.writeHDF5(pmem_test, pred_path, 'data')

    return pmem_test
