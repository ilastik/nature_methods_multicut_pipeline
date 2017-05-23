from regression_test_utils import init_train
from regression_test_utils import run_mc, run_lmc, regression_test

from multicut_src import ExperimentSettings
from multicut_src import load_dataset

import os


#
# Reference values from LCC branch
#

# vi-split, vi-merge, adapted rand
reference_values_mc = {
    'sampleA_0_train': (0.6, 0.5, 0.11),  # 0.4018, 0.3106, 0.0951
    'sampleA_1_train': (0.5, 0.5, 0.12),  # 0.3884, 0.3404, 0.1040
    'sampleB_0_train': (1.1, 1.1, 0.33),  # 0.9968, 0.9271, 0.3148
    'sampleB_1_train': (0.8, 0.8, 0.15),  # 0.6517, 0.6571, 0.1329
    'sampleC_0_train': (1.1, 0.5, 0.20),  # 0.9582, 0.3543, 0.1855
    'sampleC_1_train': (1.2, 0.6, 0.175),  # 0.9935, 0.4526, 0.1582
}

reference_values_lmc = {
    'sampleA_0_train': (0.6, 0.4, 0.11),  # 0.3953, 0.3188, 0.0963
    'sampleA_1_train': (0.5, 0.5, 0.12),  # 0.3835, 0.3404, 0.0994
    'sampleB_0_train': (1.2, 0.9, 0.25),  # 1.004, 0.7219, 0.2277
    'sampleB_1_train': (0.8, 0.6, 0.08),  # 0.6362, 0.4594, 0.0688
    'sampleC_0_train': (1.1, 0.5, 0.20),  # 0.9582, 0.3543, 0.1855
    'sampleC_1_train': (1.1, 0.5, 0.16),  # 0.9661, 0.3917, 0.1454
}


def init_regression_test_cremi(cache_folder, top_folder):
    for sample in ('A', 'B', 'C'):
        for postfix in (0, 1):
            sample_name = 'sample%s_%i' % (sample, postfix)
            data_folder = os.path.join(top_folder, sample_name)
            init_train(cache_folder, data_folder, sample_name)


def regression_test_cremi(cache_folder, top_folder, with_lmc=True):

    if not os.path.exists(os.path.join(cache_folder, 'sampleA_0_train')):
        init_regression_test_cremi(cache_folder, top_folder)

    # set the parameters
    params = ExperimentSettings()
    params.rf_cache_folder = os.path.join(cache_folder, "rf_cache")
    params.use_2d = True
    params.anisotropy_factor = 25.
    params.ignore_mask = False
    params.n_trees = 500
    params.weighting_scheme = "z"
    params.solver = "multicut_fusionmoves"
    params.verbose = True
    params.lifted_neighborhood = 3
    params.use_2rfs = True

    feature_list = ('raw', 'prob', 'reg')
    lifted_feature_list = ('cluster', 'reg')

    # run all multicuts
    mc_results  = {}
    lmc_results = {}
    for sample in ('A', 'B', 'C'):
        for postfix in (0, 1):
            ds_test = 'sample%s_%i_train' % (sample, postfix)
            train_names = [
                'sample%s_%i_train' % (train_sample, train_post) for train_sample in ('A', 'B', 'C')
                for train_post in (0, 1) if 'sample%s_%i_train' % (train_sample, train_post) != ds_test
            ]

            # print ds_test
            # print train_names

            trainsets   = [load_dataset(cache_folder, ds) for ds in train_names]
            mc_results[ds_test] = run_mc(
                trainsets,
                load_dataset(cache_folder, ds_test),
                feature_list
            )

            # run lmc if we test for it too
            if with_lmc:
                lmc_results[ds_test] = run_lmc(
                    trainsets,
                    load_dataset(cache_folder, ds_test),
                    feature_list,
                    lifted_feature_list,
                    gamma=2.
                )

    print "Eval Cremi"
    for sample in ('A', 'B', 'C'):
        for postfix in (0, 1):
            ds_test = 'sample%s_%i_train' % (sample, postfix)
            ds      = load_dataset(cache_folder, ds_test)
            gt = ds.gt()
            mc_seg = mc_results[ds_test]

            vi_split_ref, vi_merge_ref, adapted_ri_ref = reference_values_mc[ds_test]

            print "Regression Test MC for %s..." % ds_test
            regression_test(
                gt,
                mc_seg,
                vi_split_ref,
                vi_merge_ref,
                adapted_ri_ref
            )

            if with_lmc:
                print "Regression Test LMC for %s..." % ds_test
                lmc_seg = lmc_results[ds_test]
                vi_split_ref, vi_merge_ref, adapted_ri_ref = reference_values_lmc[ds_test]
                regression_test(
                    gt,
                    lmc_seg,
                    vi_split_ref,
                    vi_merge_ref,
                    adapted_ri_ref
                )


if __name__ == '__main__':
    regression_test_cremi(
        '/home/constantin/Work/home_hdd/cache/regression_tests_nfb',
        '/home/constantin/Work/neurodata_hdd/regression_test_data/cremi/cremi_transposed',
        True
    )
