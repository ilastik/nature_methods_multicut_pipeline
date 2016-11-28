import vigra
import numpy as np
import logging

from ExperimentSettings import ExperimentSettings
from EdgeRF import learn_and_predict_rf_from_gt

from init_exp import meta

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier


def run_single_rf(features_train, features_test, labels, eind1 = None, eind2 = None):
    rf = RandomForestClassifier(n_jobs = 20, n_estimators = 500)
    rf.fit(features_train, labels)
    return rf.predict_proba(features_test)


def run_two_rf(features_train_xy, features_train_z, labels,
        features_test_xy, features_test_z, edge_indications_train, edge_indications_test):

    rf_xy = RandomForestClassifier(n_jobs = 20, n_estimators = 500)
    rf_xy.fit(features_train_xy, labels[edge_indications_train == 1] )
    pmem_xy = rf_xy.predict_proba(features_test_xy)

    rf_z = RandomForestClassifier(n_jobs = 20, n_estimators = 500)
    rf_z.fit(features_train_z, labels[edge_indications_train == 0] )
    pmem_z = rf_z.predict_proba(features_test_z)

    pmem = np.zeros( (edge_indications_test.shape[0],2) )
    pmem[edge_indications_test == 1] = pmem_xy
    pmem[edge_indications_test == 0] = pmem_z

    return pmem


def cv_tworf(ds, featnames_xy, featnames_z, seg_id = 0):

    features_xy = ds.local_feature_aggregator(seg_id, featnames_xy, 5., False)
    features_z = ds.local_feature_aggregator(seg_id, featnames_z, 5., True)
    labels = ds.edge_gt(seg_id)

    # set ignore mask to 0.5
    ignore_mask = ds.ignore2ignorers(seg_id)
    assert ignore_mask.shape[0] == labels.shape[0]
    labels[ np.logical_not(ignore_mask) ] = 0.5
    labeled = labels != 0.5

    features_xy = features_xy[labeled]
    features_z = features_z[labeled]
    labels   = labels[labeled]

    assert all( np.unique(labels) == np.array([0, 1]) ), "Unique labels: " + str(np.unique(labels))

    KF = KFold(labels.shape[0], n_folds = 10)
    accuracies = []
    accuracies_xy = []
    accuracies_z  = []
    edge_ind = ds.edge_indications(seg_id)
    edge_ind = edge_ind[labeled]

    i = 1
    for train, test in KF:
        print "CV", i, "/ 10"

        f_xy_train = features_xy[train[edge_ind[train]==1]]
        f_z_train = features_z[train[edge_ind[train]==0]]

        f_xy_test = features_xy[test[edge_ind[test]==1]]
        f_z_test = features_z[test[edge_ind[test]==0]]

        pmem = run_two_rf(f_xy_train, f_z_train, labels[train],
                f_xy_test, f_z_test,
                edge_ind[train], edge_ind[test])[:,1]

        pmem[pmem>0.5] = 1
        pmem[pmem<=0.5] = 0

        accuracies.append( np.sum(pmem == labels[test]) / float( pmem.shape[0]) )

        pmem_xy = pmem[edge_ind[test]==1]
        pmem_z  = pmem[edge_ind[test]==0]
        labels_xy = labels[test][edge_ind[test]==1]
        labels_z  = labels[test][edge_ind[test]==0]

        accuracies_xy.append( np.sum(pmem_xy == labels_xy) / float( pmem_xy.shape[0]) )
        accuracies_z.append( np.sum(pmem_z == labels_z) / float( pmem_z.shape[0]) )

        i += 1

    mean = np.mean(accuracies)
    std  = np.std(accuracies)
    logging.info("accuracy mean:", mean)
    logging.info("accuracy std:", std)

    mean_xy = np.mean(accuracies_xy)
    std_xy  = np.std(accuracies_xy)

    mean_z = np.mean(accuracies_z)
    std_z  = np.std(accuracies_z)

    return mean, std, mean_xy, std_xy, mean_z, std_z


def cv_singlerf(ds, featnames, seg_id = 0):

    features = ds.local_feature_aggregator(seg_id, featnames, 5., True)
    labels = ds.edge_gt(seg_id)

    # set ignore mask to 0.5
    ignore_mask = ds.ignore2ignorers(seg_id)
    assert ignore_mask.shape[0] == labels.shape[0]
    labels[ np.logical_not(ignore_mask) ] = 0.5
    labeled = labels != 0.5

    features = features[labeled]
    labels   = labels[labeled]

    assert features.shape[0] == labels.shape[0]
    assert all( np.unique(labels) == np.array([0, 1]) ), "Unique labels: " + str(np.unique(labels))

    KF = KFold(labels.shape[0], n_folds = 10)
    accuracies = []
    accuracies_xy = []
    accuracies_z  = []
    edge_ind = ds.edge_indications(seg_id)
    edge_ind = edge_ind[labeled]

    assert edge_ind.shape[0] == features.shape[0]

    i = 1
    for train, test in KF:
        print "CV", i, "/ 10"

        pmem = run_single_rf(features[train], features[test], labels[train],
                edge_ind[train], edge_ind[test])[:,1]
        pmem[pmem>0.5] = 1
        pmem[pmem<=0.5] = 0

        accuracies.append( np.sum(pmem == labels[test]) / float( pmem.shape[0]) )

        pmem_xy = pmem[edge_ind[test]==1]
        pmem_z  = pmem[edge_ind[test]==0]
        labels_xy = labels[test][edge_ind[test]==1]
        labels_z  = labels[test][edge_ind[test]==0]

        accuracies_xy.append( np.sum(pmem_xy == labels_xy) / float( pmem_xy.shape[0]) )
        accuracies_z.append( np.sum(pmem_z == labels_z) / float( pmem_z.shape[0]) )

        i += 1

    mean = np.mean(accuracies)
    std  = np.std(accuracies)
    logging.info("accuracy mean:", mean)
    logging.info("accuracy std:", std)

    mean_xy = np.mean(accuracies_xy)
    std_xy  = np.std(accuracies_xy)

    mean_z = np.mean(accuracies_z)
    std_z  = np.std(accuracies_z)

    return mean, std, mean_xy, std_xy, mean_z, std_z



if __name__ == '__main__':

    meta.load()
    ds = meta.get_dataset("snemi3d_train")

    local_feats_list_xy = ("prob", "curve")
    local_feats_list_z  = ("prob",)

    #print "Results single RF"
    #print cv_singlerf(ds, local_feats_list_z, 1)

    print "Results two RF"
    print cv_tworf(ds, local_feats_list_xy, local_feats_list_z, 1)
