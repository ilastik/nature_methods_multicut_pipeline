import vigra
import numpy as np
import os

def view_res(res):
    from volumina_viewer import volumina_n_layer

    raw = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/neuroproof_data/raw_test.h5", "data")
    res = vigra.readHDF5(res, "data").astype(np.uint32)
    gt = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/neuroproof_data/gt_test.h5","data").astype(np.uint32)

    volumina_n_layer([raw, res, gt])


def eval_cremi(res):

    from cremi.evaluation import voi, adapted_rand

    res = vigra.readHDF5(res, "data").astype(np.uint32)
    gt = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/neuroproof_data/gt_test.h5","data").astype(np.uint32)

    #ri = adapted_rand(gt, res)
    vi_split, vi_merge = voi(gt, res)

    #print "RandScore:", 1. - ri
    #print
    print "VI-Split:", vi_split
    print "VI-Merge:", vi_merge


def eval_all(res):

    from NeuroMetrics import Metrics

    m = Metrics()

    res = vigra.readHDF5(res, "data").astype(np.uint32)
    gt = vigra.readHDF5("/home/constantin/Work/neurodata_hdd/neuroproof_data/gt_test.h5","data").astype(np.uint32)

    m.computeContingencyTable( gt.ravel(), res.ravel() )

    #print "RI", m.randIndex()
    #print "VI", m.variationOfInformation()

    print
    #print "RandScore:", m.randScore()
    print "RandRecall:", m.randRecall()
    print "RandPrecision:", m.randPrecision()

    print
    #print "ViScore:", m.viScore()
    print "ViRecall:",    m.viRecall()
    print "ViPrecision:", m.viPrecision()



if __name__ == '__main__':
    res_folder = "/home/constantin/Work/home_hdd/results/nature_results/rebuttal/neuroproof/"

    for res_name in ("res_mc.h5", "res_lmc_2.000000.h5"):

        print res_name
        res = os.path.join(res_folder, res_name)
        assert os.path.exists(res), res

        print "Scores"
        eval_all(res)

        #print
        #eval_cremi(res)

        #print
