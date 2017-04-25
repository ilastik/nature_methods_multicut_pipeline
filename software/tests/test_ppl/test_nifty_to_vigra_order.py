import numpy as np

from multicut_src import load_dataset
from init_test import meta_folder

def test_nifty_to_vigra_order():
    ds = load_dataset(meta_folder, 'test')
    seg_id = 0
    ni_2_vi = ds.nifty_to_vigra(seg_id)
    vi_2_ni = ds.vigra_to_nifty(seg_id)

    # sanity checks
    assert len(ni_2_vi) == len(vi_2_ni)

    # very simple test
    orig = np.arange( len(ni_2_vi)  )
    reordered = orig[ni_2_vi]
    back = reordered[vi_2_ni]
    assert (orig == back).all()
    print "Passed"


if __name__ == '__main__':
    test_nifty_to_vigra_order()
