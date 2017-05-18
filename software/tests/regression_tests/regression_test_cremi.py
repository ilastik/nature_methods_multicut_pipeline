from regression_test_utils import init_train, init_test
from regression_test_utils import run_mc, run_lmc, regression_test

from multicut_src import ExperimentSettings
from multicut_src import load_dataset

import vigra
import os


#
# The reference values are from Julian's experiments (with buggy LMC....)
# At some point this should def. be replaced, but for now this should be fine
#

# vi-split, vi-merge, adapted rand
reference_values = {
    'sampleA_0_train' : (0.5, 0.4, 0.11), # 0.4111, 0.3079, 0.0957
    'sampleA_1_train' : (0.5, 0.4, 0.11), # 0.3855, 0.3367, 0.0986
    'sampleB_0_train' : (1.2, 0.8, 0.24), # 1.0998, 0.7043, 0.2277
    'sampleB_1_train' : (0.8, 0.5, 0.09), # 0.6790, 0.4464, 0.0737
    'sampleC_0_train' : (1.2, 0.6, 0.23), # 1.0631, 0.4853, 0.2176
    'sampleC_1_train' : (0.5, 0.4, 0.09), # 1.3035, 0.4958, 0.2759
}


def init_regression_test_cremi(cache_folder, top_folder):
    for sample in ('A','B','C'):
        for postfix in (0,1):
            sample_name = 'sample%s_%i' % (sample, postfix)
            data_folder = os.path.join(top_folder, sample_name)
            init_train(cache_folder, data_folder, sample_name)


def regression_test_cremi(cache_folder, top_folder, with_lmc = True):

    if not os.path.exists( os.path.join(cache_folder, 'sampleA_0_train') ):
        init_regression_test_cremi(cache_folder, top_folder)

    # set the parameters
    params = ExperimentSettings()
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

    # run all multicuts
    mc_results = {}
    for sample in ('A','B','C'):
        for postfix in (0,1):
            ds_test = 'sample%s_%i_train' % (sample, postfix)
            train_names = ['sample%s_%i_train' % (train_sample, train_post) for train_sample in ('A','B','C') \
                    for train_post in (0,1) if 'sample%s_%i_train' % (train_sample, train_post) != ds_test]

            #print ds_test
            #print train_names

            trainsets   = [load_dataset(cache_folder, ds) for ds in train_names]
            mc_results[ds_test] = run_mc(
                    trainsets,
                    load_dataset(cache_folder,ds_test),
                    feature_list)

            # run lmc if we test for it too
            if with_lmc:
                lmc_results[ds_test] = run_lmc(trainsets,
                        load_dataset(cache_folder,ds_test),
                        feature_list,
                        lifted_feature_list)

    print "Eval Cremi"
    for sample in ('A','B','C'):
        for postfix in (0,1):
            ds_test = 'sample%s_%i_train' % (sample, postfix)
            ds      = load_dataset(cache_folder, ds_test)
            gt = ds.gt()
            mc_seg = mc_results[ds_test]

            vi_split_ref, vi_merge_ref, adapted_ri_ref = reference_values[ds_test]

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
                regression_test(
                        gt,
                        lmc_seg,
                        vi_split_ref,
                        vi_merge_ref,
                        adapted_ri_ref
                        )



if __name__ == '__main__':
    regression_test_cremi(
            '/home/constantin/Work/home_hdd/cache/regression_tests_lcc',
            '/home/constantin/Work/neurodata_hdd/regression_test_data/cremi/cremi',
            True
            )
