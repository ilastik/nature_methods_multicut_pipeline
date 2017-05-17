import os
import vigra
from volumina_viewer import volumina_n_layer

def view_nproof(cache_folder, data_folder):
    pass


def view_isbi(cache_folder, data_folder):
    raw  = vigra.readHDF5( os.path.join(cache_folder, 'isbi_test/inp0.h5'), 'data')
    pmap = vigra.readHDF5( os.path.join(cache_folder, 'isbi_test/inp1.h5'), 'data')
    seg  = vigra.readHDF5( os.path.join(cache_folder, 'isbi_test/seg0.h5'), 'data')

    seg_mc     = vigra.readHDF5( os.path.join(cache_folder, 'isbi_test/mc_seg.h5'), 'data' )
    seg_ref_mc = vigra.readHDF5( os.path.join(data_folder, 'isbi_transposed/mc_seg.h5'), 'data' )

    seg_lmc     = vigra.readHDF5(os.path.join(cache_folder, 'isbi_test/lmc_seg.h5'), 'data' )
    seg_ref_lmc = vigra.readHDF5(os.path.join(data_folder, 'isbi_transposed/lmc_seg.h5'), 'data' )

    #volumina_n_layer(
    #        [raw,   pmap,    seg,  seg_mc,   seg_ref_mc],
    #        ['raw', 'pmap', 'seg','seg_mc', 'seg_ref_mc']
    #        )

    volumina_n_layer(
            [raw, seg_mc, seg_ref_mc, seg_lmc, seg_ref_lmc],
            ['raw', 'seg_mc', 'seg_ref_mc', 'seg_lmc', 'seg_ref_lmc']
            )

def view_isbi_train():
    raw  = vigra.readHDF5('./cache_isbi/isbi_train/inp0.h5', 'data')
    pmap = vigra.readHDF5('./cache_isbi/isbi_train/inp1.h5', 'data')
    seg  = vigra.readHDF5('./cache_isbi/isbi_train/seg0.h5', 'data')
    gt   = vigra.readHDF5('./cache_isbi/isbi_train/gt.h5', 'data')
    volumina_n_layer(
            [raw,   pmap,    seg,  gt ],
            ['raw', 'pmap', 'seg', 'gt']
            )

if __name__ == '__main__':
    view_isbi(
            '/home/constantin/Work/home_hdd/cache/regression_tests_nfb',
            '/home/constantin/Work/neurodata_hdd/regression_test_data')
