from skneuro.learning import randIndex, variationOfInformation
import vigra
import numpy as np

def eval_neuroproof(res):
    gt = vigra.readHDF5("/home/constantin/home_hdd/data/neuroproof_data/gt_test.h5",
            "data").astype(np.uint32)

    ri = randIndex(gt.ravel(), res.ravel(), ignoreDefaultLabel = False)
    vi = variationOfInformation(gt.ravel(), res.ravel(), ignoreDefaultLabel = False)

    print "RI:", ri
    print "VI:", vi


if __name__ == '__main__':
    res = vigra.readHDF5("./neuroproof_lmcresult.h5", "data")
    print eval_neuroproof(res)
