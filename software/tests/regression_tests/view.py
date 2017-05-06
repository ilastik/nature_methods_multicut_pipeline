import os
import vigra
from volumina_viewer import volumina_n_layer

def view_isbi():
    raw  = vigra.readHDF5('./cache_isbi/isbi_test/inp0.h5', 'data')
    pmap = vigra.readHDF5('./cache_isbi/isbi_test/inp1.h5', 'data')
    seg  = vigra.readHDF5('./cache_isbi/isbi_test/seg0.h5', 'data')

    seg_mc = vigra.readHDF5('./cache_isbi/isbi_test/mc_seg.h5', 'data')
    seg_ref_mc = vigra.readHDF5('./data/isbi/mc_seg.h5', 'data')

    #seg_lmc = vigra.readHDF5('./cache_isbi/isbi_test/lmc_seg.h5', 'data')
    #seg_ref_lmc = vigra.readHDF5('./data/isbi/lmc_seg.h5', 'data')

    volumina_n_layer(
            [raw,   pmap,    seg,  seg_mc,   seg_ref_mc],
            ['raw', 'pmap', 'seg','seg_mc', 'seg_ref_mc']
            )

    #volumina_n_layer(
    #        [raw, seg_mc, seg_ref_mc, seg_lmc, seg_ref_lmc],
    #        ['raw', 'seg_mc', 'seg_ref_mc', 'seg_lmc', 'seg_ref_lmc']
    #        )


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
    view_isbi_train()
