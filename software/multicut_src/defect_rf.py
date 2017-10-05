from __future__ import print_function, division

import h5py
import numpy as np
import vigra

from .EdgeRF import RandomForest
from .DataSet import Cutout
from .ExperimentSettings import ExperimentSettings
from .tools import cacher_hdf5

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


# TODO cacher
def defect_node_gt(ds, seg_id, defect_gt_path, defect_gt_key):

    # load the defect ground-truth
    with h5py.File(defect_gt_path) as f:
        # if we have a cutout, get the defect gt from the bounding box
        if isinstance(ds, Cutout):
            bb = ds.bb
            defect_gt = f[defect_gt_key][bb]
        else:
            defect_gt = f[defect_gt_key][:]

    # map the defect gt to node labels
    rag = ds.rag(seg_id)
    node_labels = nrag.gridRagAccumulateLabels(rag, defect_gt)
    return node_labels


# get region statistics with the vigra region feature extractor
@cacher_hdf5(folder="feature_folder")
def node_features(ds, seg_id, inp_id):
    assert seg_id < ds.n_seg, str(seg_id) + " , " + str(ds.n_seg)
    assert inp_id < ds.n_inp, str(inp_id) + " , " + str(ds.n_inp)

    # list of the region statistics, that we want to extract
    statistics = ["Mean", "Count", "Kurtosis",
                  "Maximum", "Minimum", "Quantiles",
                  "RegionRadii", "Skewness", "Sum",
                  "Histogram" "Variance"]

    extractor = vigra.analysis.extractRegionFeatures(
        ds.inp(inp_id).astype("float32"),
        ds.seg(seg_id).astype("uint32"),
        features=statistics
    )

    node_features = np.concatenate(
        [extractor[stat_name][:, None].astype('float32') if extractor[stat_name].ndim == 1
            else extractor[stat_name].astype('float32') for stat_name in statistics],
        axis=1
    )
    return node_features


def learn_and_predict_defect_rf(
    trainsets,
    ds_test,
    seg_id_train,
    seg_id_test,
    defect_gt_paths,
    defect_gt_keys
):
    # for only a single ds, put it in a list
    if not isinstance(trainsets, (list, tuple)):
        trainsets = [trainsets]

    # strings for caching
    # str for all relevant params
    paramstr = "_".join(
        [str(ExperimentSettings().n_trees), '_'.join(defect_gt_paths).replace('/', '')]
    )
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    trainstr = "_".join([ds.ds_name for ds in trainsets]) + "_" + str(seg_id_train)

    # we cache this in the ds_test cache folder
    # if caching random forests is activated (== rf_cache_folder is not None)
    if ExperimentSettings().rf_cache_folder is not None:  # cache-folder exists => look if we already have a prediction

        pred_name = "defect_prediction_" + "_".join([trainstr, teststr, paramstr]) + ".h5"
        if len(pred_name) > 255:
            pred_name = str(hash(pred_name[:-3])) + ".h5"
        pred_path = os.path.join(ds_test.cache_folder, pred_name)

        # see if the rf is already learned and predicted, otherwise learn it
        if os.path.exists(pred_path):
            return vigra.readHDF5(pred_path, 'data')

    rf = learn_defect_rf(trainsets, seg_id_train, defect_gt_paths, defect_gt_keys)
    features = node_features(ds_test, seg_id_test, 0)

    p_defect = rf.predict_probabilities(features.astype('float32'))[:, 1]

    if ExperimentSettings().rf_cache_folder is not None:
        vigra.writeHDF5(p_defect, pred_path, 'data')

    return p_defect


def learn_defect_rf(
    trainsets,
    seg_id,
    trainstr,
    paramstr,
    defect_gt_paths,
    defect_gt_keys
):

    # check if this random forest is already cached
    cache_folder = ExperimentSettings().rf_cache_folder
    if cache_folder is not None:  # we use caching for the rf => look if already exists
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        rf_folder = os.path.join(cache_folder, "defect_rf_" + trainstr)
        rf_name = "defect_rf_" + "_".join([trainstr, paramstr])
        if len(rf_name) > 255:
            rf_name = str(hash(rf_name))

        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)
        rf_path   = os.path.join(rf_folder, rf_name)
        if RandomForest.is_cached(rf_path):
            return RandomForest.load_from_file(rf_path, 'rf', ExperimentSettings().n_threads)

    # iterate over the training sets and calculate features and labels
    features = []
    labels = []

    for i, ds in enumerate(trainsets):
       features.append(node_features(ds, seg_id, 0))
       labels.append(defect_node_gt(ds, seg_id, defect_gt_paths[i], defect_gt_keys[i]))

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    assert len(features) == len(labels), "%i, %i" % (len(features), len(labels))

    rf = RandomForest(
        features.astype('float32'),
        labels,
        n_trees=ExperimentSettings().n_trees,
        n_threads=ExperimentSettings().n_threads
    )

    if cache_folder is not None:
        rf.write(rf_path, 'rf')

    return rf
