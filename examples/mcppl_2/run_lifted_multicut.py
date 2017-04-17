import vigra
import os

from init_exp import meta_folder

from multicut_src import lifted_multicut_workflow, lifted_multicut_workflow_with_defect_correction
from multicut_src import ExperimentSettings, load_dataset

def run_lmc(ds_train_name, ds_test_name, mc_params, save_path):

    assert os.path.exists(os.path.split(save_path)[0]), "Please choose an existing folder to save your results"

    # if you have added multiple segmentations, you can choose on which one to run
    # experiments with the seg_id
    seg_id = 0

    # these strings encode the features that are used for the local features
    feature_list = ['raw', 'prob', 'reg']

    # these strings encode the features that will be used for the lifted edges
    feature_list_lifted = ['cluster', 'reg']

    # this factor determines the weighting of lifted vs. local edge costs
    gamma = 2.

    ds_train = load_(meta_folder, ds_train_name)
    ds_test  = load_(meta_folder, ds_test_name)

    # need to make filters for the trainset beforehand
    ds_train.make_filters(0, mc_params.anisotropy_factor)
    ds_train.make_filters(1, mc_params.anisotropy_factor)

    # use this for running the mc without defected slices
    mc_nodes, _, _, _ = lifted_multicut_workflow(
            ds_train, ds_test,
            seg_id, seg_id,
            feature_list, feature_list_lifted,
            mc_params, gamma = gamma)

    # use this for running the mc with defected slices
    #mc_nodes, _, _, _ = lifted_multicut_workflow_with_defect_correction(
    #        ds_train, ds_test,
    #        seg_id, seg_id,
    #        feature_list, feature_list_lifted,
    #        mc_params, gamma = gamma)

    segmentation = ds_test.project_mc_result(seg_id, mc_nodes)
    vigra.writeHDF5(segmentation, save_path, 'data', compression = 'gzip')

if __name__ == '__main__':

    # this object stores different  experiment settings
    mc_params = ExperimentSettings()
    mc_params.set_rfcache(meta_folder, "rf_cache")

    anisotropy = 25. # the anisotropy of the data, this is used in the filter calculation
    # set to 1. for isotropic data, to the actual degree for mildly anisotropic data or to > 20. to compute filters in 2d
    mc_params.set_anisotropy(25.)

    # set to true for segmentations with flat superpixels
    mc_params.set_use2d(True)

    # number of threads used for multithreaded computations
    mc_params.set_nthreads(8)

    # number of trees used in the random forest
    mc_params.set_ntrees(200)

    # solver used for the multicut
    mc_params.set_solver("multicut_fusionmoves")
    # pipeline verbosity
    mc_params.set_verbose(True)

    # weighting scheme for edge-costs in the mc problem
    # set to 'none' for no weighting
    # 'z' or 'xyz' or 'all' for flat superpixels (z usually works best)
    # 'all' for isotropic data with 3d superpixel
    mc_params.set_weighting_scheme("z")

    # range of lifted edges
    mc_params.set_lifted_neighborhood(3)

    # path to save the segmentation result, order has to already exist
    save_path = '/path/to/mc_result.h5'
    run_lmc('my_train', 'my_test', mc_params, save_path)
