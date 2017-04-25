import numpy as np

from multicut_src import load_dataset
from init_test_aniso import meta_folder

def test_rags():
    ds = load_dataset(meta_folder, 'test')
    seg_id = 0
    rag_vi = ds._rag(seg_id)
    rag_ni = ds.nifty_rag(seg_id)
    assert rag_vi.nodeNum == rag_ni.numberOfNodes
    assert rag_vi.edgeNum == rag_ni.numberOfEdges
    print "Passed rag test"

def test_nifty_to_vigra_order():
    ds = load_dataset(meta_folder, 'test')
    seg_id = 0
    ni_2_vi = ds.nifty_to_vigra(seg_id)
    vi_2_ni = ds.vigra_to_nifty(seg_id)

    rag_vi = ds._rag(seg_id)
    rag_ni = ds.nifty_rag(seg_id)

    uv_ni = rag_ni.uvIds()
    uv_vi = np.sort( rag_vi.uvIds(), axis = 1)

    # sanity checks
    assert len(ni_2_vi) == len(vi_2_ni)

    # very simple test
    orig = np.arange( len(ni_2_vi)  )
    reordered = orig[ni_2_vi]
    back = reordered[vi_2_ni]
    compare = (orig == back)

    print uv_ni[compare]
    print uv_vi[compare]

    assert compare.all(), "%i / %i" % (np.sum(compare), len(compare))
    print "Passed"


if __name__ == '__main__':
    test_rags()
    test_nifty_to_vigra_order()
