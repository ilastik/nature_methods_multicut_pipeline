import os
import vigra

from multicut_src import DataSet
from multicut_src import multicut_workflow, lifted_multicut_workflow

from cremi.evaluation import NeuronIds
from cremi import Volume


def init_train(cache_folder, data_folder, ds_prefix):

    ds_train = DataSet(cache_folder, '%s_train' % ds_prefix)
    ds_train.add_raw(os.path.join(data_folder, '%s_raw_train.h5' % ds_prefix), 'data')
    ds_train.add_input(os.path.join(data_folder, '%s_pmap_train.h5' % ds_prefix), 'data')
    ds_train.add_seg(os.path.join(data_folder, '%s_seg_train.h5' % ds_prefix), 'data')
    ds_train.add_gt(os.path.join(data_folder, '%s_gt_train.h5' % ds_prefix), 'data')

    shape = ds_train.shape
    z0 = 0
    z1 = int(shape[0] * 0.2)
    z2 = int(shape[0] * 0.8)
    z3 = shape[0]

    ds_train.make_cutout([z0, 0, 0], [z1, shape[1], shape[2]])
    ds_train.make_cutout([z1, 0, 0], [z2, shape[1], shape[2]])
    ds_train.make_cutout([z2, 0, 0], [z3, shape[1], shape[2]])


def init_test(cache_folder, data_folder, ds_prefix):

    ds_test = DataSet(cache_folder, '%s_test' % ds_prefix)
    ds_test.add_raw(os.path.join(data_folder, '%s_raw_test.h5' % ds_prefix), 'data')
    ds_test.add_input(os.path.join(data_folder, '%s_pmap_test.h5' % ds_prefix), 'data')
    ds_test.add_seg(os.path.join(data_folder, '%s_seg_test.h5' % ds_prefix), 'data')


def init(cache_folder, data_folder, ds_prefix):
    init_train(cache_folder, data_folder, ds_prefix)
    init_test(cache_folder, data_folder, ds_prefix)


def run_mc(ds_train, ds_test, feature_list):
    mc_nodes, _, _, _ = multicut_workflow(
        ds_train, ds_test,
        0, 0,
        feature_list)
    return ds_test.project_mc_result(0, mc_nodes)


def run_lmc(ds_train, ds_test, local_feature_list, lifted_feature_list, gamma):
    mc_nodes, _, _, _ = lifted_multicut_workflow(
        ds_train, ds_test,
        0, 0,
        local_feature_list,
        lifted_feature_list)
    return ds_test.project_mc_result(0, mc_nodes)


def evaluate(gt, segmentation):
    gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=1)
    evaluate = NeuronIds(Volume(gt))

    segmentation = Volume(segmentation)
    vi_split, vi_merge = evaluate.voi(segmentation)
    ri = evaluate.adapted_rand(segmentation)

    return vi_split, vi_merge, ri


def regression_test(
        ref_seg,
        seg,
        expected_vi_split=0,
        expected_vi_merge=0,
        expected_ri=0
):
    vi_split, vi_merge, ri = evaluate(ref_seg, seg)
    vi_s_pass = vi_split < expected_vi_split
    vi_m_pass = vi_merge < expected_vi_merge
    ri_pass   = ri < expected_ri
    if vi_m_pass and vi_s_pass and ri_pass:
        print "All passed with:"
        print "Vi-Split:", vi_split, "(Ref:)", expected_vi_split
        print "Vi-Merge:", vi_merge, "(Ref:)", expected_vi_merge
        print "RI:", ri, "(Ref:)", expected_ri
    else:
        print "FAILED with"
        print "Vi-Split: %s with %f, (Ref:) %f" % ('Passed' if vi_s_pass else 'FAILED', vi_split, expected_vi_split)
        print "Vi-Merge: %s with %f, (Ref:) %f" % ('Passed' if vi_m_pass else 'FAILED', vi_merge, expected_vi_merge)
        print "RI: %s with %f, (Ref:) %f" % ('Passed' if ri_pass else 'FAILED', ri, expected_ri)


def clean_up():
    pass
