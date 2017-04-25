import numpy as np

from multicut_src import load_dataset
from init_test import meta_folder_anisotropic as meta_folder

def test_affinity_features():
    ds = load_dataset(meta_folder, 'test')
    seg_id = 0

    rag = ds._rag(seg_id)
    aff0 = ds.edge_features_from_affinity_maps(seg_id, (0,1), 20., 0)
    assert aff0.shape[0] == rag.edgeNum
    aff1 = ds.edge_features_from_affinity_maps(seg_id, (0,1), 20., 1)
    assert aff1.shape == aff0.shape
    aff2 = ds.edge_features_from_affinity_maps(seg_id, (0,1), 20., 2)
    assert aff2.shape == aff0.shape

    print "Passed"


def test_edge_features():
    ds = load_dataset(meta_folder, 'test')
    seg_id = 0
    feats0 = ds.edge_features(seg_id,0,20)
    feats1 = ds.edge_features(seg_id,1,20)
    assert feats0.shape == feats1.shape


if __name__ == '__main__':
    test_edge_features()
