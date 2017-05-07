import os
import numpy as np

from multicut_src import DataSet
from multicut_src import MetaSet
#from multicut_src import load_dataset

from multicut_src import multicut_workflow, lifted_multicut_workflow

from cremi.evaluation import NeuronIds
from cremi import Volume


def init(cache_folder, data_folder, ds_prefix):

    meta = MetaSet(cache_folder)

    ds_train = DataSet(cache_folder, '%s_train' % ds_prefix)
    ds_train.add_raw(  os.path.join( data_folder, 'raw_train.h5' ), 'data')
    ds_train.add_input(os.path.join( data_folder, 'pmap_train.h5'), 'data')
    ds_train.add_seg(  os.path.join( data_folder, 'seg_train.h5' ), 'data')
    ds_train.add_gt(   os.path.join( data_folder, 'gt_train.h5'  ), 'data')

    shape = ds_train.shape
    z0 = 0
    z1 = int( shape[2] * 0.2 )
    z2 = int( shape[2] * 0.8 )
    z3 = shape[2]

    ds_train.make_cutout([0, shape[0], 0, shape[1], z0, z1])
    ds_train.make_cutout([0, shape[0], 0, shape[1], z1, z2])
    ds_train.make_cutout([0, shape[0], 0, shape[1], z2, z3])

    ds_test = DataSet(cache_folder, '%s_test' % ds_prefix)
    ds_test.add_raw(  os.path.join(data_folder, 'raw_test.h5' ), 'data')
    ds_test.add_input(os.path.join(data_folder, 'pmap_test.h5'), 'data')
    ds_test.add_seg(  os.path.join(data_folder, 'seg_test.h5' ), 'data')

    meta.add_dataset('%s_train' % ds_prefix, ds_train)
    meta.add_dataset('%s_test'  % ds_prefix, ds_test)
    meta.save()

    return meta


def run_mc(ds_train, ds_test, feature_list, mc_params):
    mc_nodes, _, _, _ = multicut_workflow(
            ds_train, ds_test,
            0, 0,
            feature_list, mc_params)
    return ds_test.project_mc_result(0, mc_nodes)


def run_lmc(ds_train, ds_test, local_feature_list, lifted_feature_list, mc_params, gamma):
    mc_nodes, _, _, _ = lifted_multicut_workflow(
            ds_train, ds_test,
            0, 0,
            local_feature_list, lifted_feature_list,
            mc_params)
    return ds_test.project_mc_result(0, mc_nodes)


def evaluate(gt, segmentation):
    evaluate = NeuronIds( Volume(gt) )

    segmentation = Volume(segmentation)
    vi_split, vi_merge = evaluate.voi(segmentation)
    ri = evaluate.adapted_rand(segmentation)

    return vi_split, vi_merge, ri


def regression_test(
        ref_seg,
        seg,
        expected_vi_split = 0,
        expected_vi_merge = 0,
        expected_ri       = 0
        ):
    vi_split, vi_merge, ri = evaluate(ref_seg, seg)
    print vi_split, vi_merge, ri
    assert vi_split < expected_vi_split, "%f, %f" % (vi_split, expected_vi_split)
    assert vi_merge < expected_vi_merge, "%f, %f" % (vi_merge, expected_vi_split)
    assert ri < expected_ri, "%f, %f" % (ri, expected_ri)
    print "Passed with:"
    print "Vi-Split:", vi_split, "(Ref:)", expected_vi_split
    print "Vi-Merge:", vi_merge, "(Ref:)", expected_vi_merge
    print "RI::", ri, "(Ref:)", expected_ri


def clean_up():
    pass
