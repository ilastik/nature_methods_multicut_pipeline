import vigra
import os
import numpy as np

def compate_caches_feats(feat_folder_1, feat_folder_2):
    feat_list_1 = os.listdir(feat_folder_1)
    feat_list_2 = os.listdir(feat_folder_2)

    # compare edge features - raw
    edge_raw_1 = [ff for ff in feat_list_1 if ff.startswith('edge_features_0_0')]
    assert len(edge_raw_1) == 1
    raw_1 = vigra.readHDF5(
            os.path.join(feat_folder_1, edge_raw_1[0]), 'data' )

    edge_raw_2 = [ff for ff in feat_list_2 if ff.startswith('edge_features_0_0')]
    assert len(edge_raw_2) == 1
    raw_2 = vigra.readHDF5(
            os.path.join(feat_folder_2, edge_raw_2[0]), 'data' )
    assert raw_1.shape == raw_2.shape, "%s, %s" % (str(raw_1.shape), str(raw_2.shape))
    assert np.allclose(raw_1, raw_2), "%i / %i" % (np.sum(np.isclose(raw_1,raw_2)), raw_1.size)

    # compare edge features - pmap
    edge_pmap_1 = [ff for ff in feat_list_1 if ff.startswith('edge_features_0_1')]
    assert len(edge_pmap_1) == 1
    pmap_1 = vigra.readHDF5(
            os.path.join(feat_folder_1, edge_pmap_1[0]), 'data' )

    edge_pmap_2 = [ff for ff in feat_list_2 if ff.startswith('edge_features_0_1')]
    assert len(edge_pmap_2) == 1
    pmap_2 = vigra.readHDF5(
            os.path.join(feat_folder_2, edge_pmap_2[0]), 'data' )
    assert pmap_1.shape == pmap_2.shape, "%s, %s" % (str(pmap_1.shape), str(pmap_2.shape))
    assert np.allclose(pmap_1, pmap_2), "%i / %i" % (np.sum(np.isclose(pmap_1,pmap_2)), pmap_1.size)

    # compare region features
    edge_region_1 = [ff for ff in feat_list_1 if ff.startswith('region_features')]
    #assert len(edge_region_1) == 2, str(len(edge_region_1))
    region_1 = vigra.readHDF5(
            os.path.join(feat_folder_1, edge_region_1[0]), 'data' )

    edge_region_2 = [ff for ff in feat_list_2 if ff.startswith('region_features')]
    #assert len(edge_region_2) == 1
    region_2 = vigra.readHDF5(
            os.path.join(feat_folder_2, edge_region_2[0]), 'data' )
    assert region_1.shape == region_2.shape, "%s, %s" % (str(region_1.shape), str(region_2.shape))
    assert np.allclose(region_1, region_2), "%i / %i" % (np.sum(np.isclose(region_1,region_2)), region_1.size)


    # compare topo features
    edge_region_1 = [ff for ff in feat_list_1 if ff.startswith('region_features')]
    #assert len(edge_region_1) == 2, str(len(edge_region_1))
    region_1 = vigra.readHDF5(
            os.path.join(feat_folder_1, edge_region_1[0]), 'data' )

    edge_region_2 = [ff for ff in feat_list_2 if ff.startswith('region_features')]
    #assert len(edge_region_2) == 1
    region_2 = vigra.readHDF5(
            os.path.join(feat_folder_2, edge_region_2[0]), 'data' )
    assert region_1.shape == region_2.shape, "%s, %s" % (str(region_1.shape), str(region_2.shape))
    assert np.allclose(region_1, region_2), "%i / %i" % (np.sum(np.isclose(region_1,region_2)), region_1.size)

    print "Feature check passed"


def compare_caches_filters(filter_folder_1, filter_folder_2):
    filters = os.listdir(filter_folder_1)

    for ff in filters:
        ff1 = os.path.join(filter_folder_1, ff)
        ff2 = os.path.join(filter_folder_2, ff) + '00000'
        assert os.path.exists(ff2), ff2
        filt1 = vigra.readHDF5(ff1, 'data')
        filt2 = vigra.readHDF5(ff2, 'data')
        assert np.allclose(filt1, filt2), "%s: %i / %i" % (ff, np.sum(np.isclose(filt1,filt2)), filt1.size)


def compare_caches_test(cache_folder_1, cache_folder_2):

    files_1 = os.listdir(cache_folder_1)
    files_2 = os.listdir(cache_folder_2)

    # compare features
    compate_caches_feats(
            os.path.join(cache_folder_1,'features'),
            os.path.join(cache_folder_2,'features'))


    # compare multicut weights
    weights_1 = [ff for ff in files_1 if ff.startswith('probs_to_energies')]
    assert len(weights_1) == 1
    mc_weigths_1 = vigra.readHDF5(
            os.path.join(cache_folder_1, weights_1[0]), 'data')

    weights_2 = [ff for ff in files_2 if ff.startswith('probs_to_energies')]
    assert len(weights_2) == 1
    mc_weigths_2 = vigra.readHDF5(
            os.path.join(cache_folder_2, weights_2[0]), 'data')
    assert np.allclose(mc_weigths_1, mc_weigths_2), "%i / %i" % (np.sum(np.isclose(mc_weigths_1,mc_weigths_2)), len(mc_weigths_1))



def compare_caches_train(cache_folder_1, cache_folder_2):

    files_1 = os.listdir(cache_folder_1)
    files_2 = os.listdir(cache_folder_2)

    # compare edge labels
    edge_labels_1 = [ff for ff in files_1 if ff.startswith('edge_gt')]
    assert len(edge_labels_1) == 1
    labels_1 = vigra.readHDF5(
            os.path.join(cache_folder_1, edge_labels_1[0]), 'data')

    edge_labels_2 = [ff for ff in files_2 if ff.startswith('edge_gt')]
    assert len(edge_labels_2) == 1
    labels_2 = vigra.readHDF5(
            os.path.join(cache_folder_2, edge_labels_2[0]), 'data')
    assert np.allclose(labels_1, labels_2), "%i / %i" % (np.sum(np.isclose(labels_1,labels_2)), len(labels_1))
    print "Passed labels check"

    # compare filters
    compare_caches_filters(
            cache_folder_1 + '/filters/filters_2d/inp_0',
            cache_folder_2 + '/filters/filters_2d/inp_0'
            )
    compare_caches_filters(
            cache_folder_1 + '/filters/filters_2d/inp_1',
            cache_folder_2 + '/filters/filters_2d/inp_1'
            )
    print "Passed filters check"

    # compare features
    compate_caches_feats(
            os.path.join(cache_folder_1,'features'),
            os.path.join(cache_folder_2,'features'))


if __name__ == '__main__':
    compare_caches_train(
            '/home/constantin/Work/home_hdd/cache/regression_tests_master/isbi_train',
            '/home/constantin/Work/home_hdd/cache/regression_tests_lcc/isbi_train'
            )
    compare_caches_test(
            '/home/constantin/Work/home_hdd/cache/regression_tests_master/isbi_test',
            '/home/constantin/Work/home_hdd/cache/regression_tests_lcc/isbi_test'
            )
