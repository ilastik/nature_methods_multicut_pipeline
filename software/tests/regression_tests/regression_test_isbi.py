import os
import vigra
import argparse
from regression_test_utils import init, run_mc, run_lmc, regression_test

from multicut_src import ExperimentSettings
from multicut_src import MetaSet
#from multicut_src import load_dataset

def regression_test_isbi(cache_folder, data_folder):

    # if the cache does not exist, create it
    if not os.path.exists( os.path.join(cache_folder,'isbi_train') ):
        meta = init(cache_folder, data_folder, 'isbi')
    else:
        meta = MetaSet(cache_folder)
        meta.load()

    # isbi params
    params = ExperimentSettings()
    params.rf_cache_folder = os.path.join(cache_folder, "rf_cache")
    params.use_2d = True
    params.anisotropy_factor = 25.
    params.learn_2d = True
    params.ignore_mask = False
    params.n_trees = 500
    params.weighting_scheme = "z"
    params.solver = "multicut_fusionmoves"

    local_feats_list  = ("raw", "prob", "reg", "topo")
    lifted_feats_list = ("mc", "cluster", "reg")

    ds_train = meta.get_dataset('isbi_train')
    ds_test  = meta.get_dataset('isbi_test')
    mc_seg  = run_mc( ds_train, ds_test, local_feats_list, params)
    lmc_seg = run_lmc(ds_train, ds_test, local_feats_list, lifted_feats_list, params, 2.)

    #vigra.writeHDF5(mc_seg, './cache_isbi/isbi_test/mc_seg.h5', 'data', compression = 'gzip')
    #vigra.writeHDF5(lmc_seg, './cache_isbi/isbi_test/lmc_seg.h5', 'data', compression = 'gzip')

    print "Regression Test MC..."
    # Eval differences with same parameters and according regression thresholds
    # vi-split:   0.0718660622942 -> 0.1
    vi_split_ref = 0.1
    # vi-merge:   0.0811051987574 -> 0.1
    vi_merge_ref = 0.1
    # adapted-ri: 0.0218391269081 -> 0.05
    adapted_ri_ref = 0.05
    regression_test(
            vigra.readHDF5(os.path.join(data_folder,'mc_seg.h5'), 'data'),
            mc_seg,
            vi_split_ref,
            vi_merge_ref,
            adapted_ri_ref
            )
    print "... passed"

    print "Regression Test LMC..."
    # Eval differences with same parameters and according regression thresholds
    # vi-split: 0.161923549092 -> 0.2
    vi_split_ref = 0.2
    # vi-merge: 0.0792288680404 -> 0.1
    vi_merge_ref = 0.1
    # adapted-ri: 0.0334914933439 -> 0.05
    adapted_ri_ref = 0.05
    regression_test(
            vigra.readHDF5(os.path.join(data_folder,'lmc_seg.h5'), 'data'),
            lmc_seg,
            vi_split_ref,
            vi_merge_ref,
            adapted_ri_ref
            )
    print "... passed"


if __name__ == '__main__':
    regression_test_isbi(
            '/home/constantin/Work/home_hdd/cache/regression_tests_master',
            '/home/constantin/Work/neurodata_hdd/regression_test_data/isbi')
