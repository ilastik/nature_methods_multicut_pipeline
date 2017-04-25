import numpy as np

from multicut_src import load_dataset
from init_test import meta_folder_isotropic, meta_folder_anisotropic

def test_rags(aniso = True):
    meta_folder = meta_folder_anisotropic if aniso else meta_folder_isotropic
    ds = load_dataset(meta_folder, 'test')
    seg_id = 0
    rag_vi = ds._rag(seg_id)
    rag_ni = ds.nifty_rag(seg_id)
    assert rag_vi.nodeNum == rag_ni.numberOfNodes
    assert rag_vi.edgeNum == rag_ni.numberOfEdges
    print "Passed rag test"

def test_nifty_to_vigra_order(aniso = True):
    meta_folder = meta_folder_anisotropic if aniso else meta_folder_isotropic
    ds = load_dataset(meta_folder, 'test')
    seg_id = 0

    ni_2_vi = ds.nifty_to_vigra(seg_id)
    vi_2_ni = ds.vigra_to_nifty(seg_id)

    rag_vi = ds._rag(seg_id)
    rag_ni = ds.nifty_rag(seg_id)

    uv_ni = rag_ni.uvIds()
    uv_vi = np.sort( rag_vi.uvIds(), axis = 1)

    assert len(ni_2_vi) == len(vi_2_ni)
    assert (uv_ni[ni_2_vi] == uv_vi).all()
    assert (uv_vi[vi_2_ni] == uv_ni).all()

    print "Passed nifty to vigra converter tests"



if __name__ == '__main__':
    test_rags()
    test_nifty_to_vigra_order()
