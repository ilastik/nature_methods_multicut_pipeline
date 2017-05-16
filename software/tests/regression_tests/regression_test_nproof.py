import os
import vigra
import argparse
from regression_test_utils import init, run_mc, run_lmc, regression_test

from multicut_src import ExperimentSettings
from multicut_src import load_dataset

def regression_test_snemi(cache_folder, data_folder):

    # if the cache does not exist, create it
    if not os.path.exists(cache_folder):
        meta = init(cache_folder, data_folder, 'nproof')
    else:
        meta = MetaSet(cache_folder)
        meta.load()

    # isbi params
    params = ExperimentSettings()
    params.rf_cache_folder = os.path.join(cache_folder, "rf_cache")
    params.use_2d = True
    params.anisotropy_factor = 1.
    params.ignore_mask = False
    params.n_trees = 500
    params.solver = "multicut_fusionmoves"
    params.lifted_neighborhood = 2

    local_feats_list  = ("raw", "prob", "reg", "topo")
    lifted_feats_list = ("cluster", "reg")

    ds_train = meta.get_dataset(cache_folder, 'nproof_train')
    ds_test  = meta.get_dataset(cache_folder, 'nproof_test')
    mc_seg  = run_mc( ds_train, ds_test, local_feats_list)
    lmc_seg = run_lmc(ds_train, ds_test, local_feats_list, lifted_feats_list, 2)

    print "Regression Test MC..."
    # Eval differences with same parameters and according regression thresholds
    # vi-split: 0.31985479849 -> 0.35
    vi_split_ref = 0.35
    # vi-merge: 0.402968960935 -> 0.45
    vi_merge_ref = 0.45
    # adapted-ri: 0.122123986224 -> 0.15
    adapted_ri_ref = 0.15
    regression_test(
            vigra.readHDF5(os.path.join(data_folder,'mc_seg.h5'), 'data'),
            mc_seg
            )
    print "... passed"

    print "Regression Test LMC..."
    # Eval differences with same parameters and according regression thresholds
    # vi-split: 0.332745302066 => 0.4
    vi_split_ref = 0.4
    # vi-merge: 0.332349723508 => 0.4
    vi_merge_ref = 0.4
    # adapted-ri: 0.0942531472586 => 0.12
    adapted_ri_ref = 0.12
    regression_test(
            vigra.readHDF5(os.path.join(data_folder,'lmc_seg.h5'), 'data'),
            lmc_seg
            )
    print "... passed"


if __name__ == '__main__':
    regression_test_isbi(
            '/home/constantin/Work/home_hdd/cache/regression_tests_nfb',
            '/home/constantin/Work/neurodata_hdd/regression_test_data/nproof')
