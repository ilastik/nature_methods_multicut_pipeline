from __future__ import print_function, division
import vigra
import numpy as np

from .ExperimentSettings import ExperimentSettings
from .lifted_mc import compute_and_save_lifted_nh
from .tools import find_matching_row_indices

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


# TODO cache
# get the long range z adjacency
def get_long_range_z_adjacency(ds, seg_id, lifted_range):
    rag = ds.rag(seg_id)
    adjacency = nrag.getLongRangeAdjacency(rag, lifted_range)
    if ds.has_seg_mask:
        where_uv = (adjacency != ExperimentSettings().ignore_seg_value).all(axis=1)
        adjacency = adjacency[where_uv]
    return adjacency


# get features of the long range z-adjacency from affinity maps
def get_long_range_z_features(ds, seg_id, affinity_map_path, affinity_map_key, lifted_range):
    affinity_maps = vigra.readHDF5(affinity_map_path, affinity_map_key)
    # this assumes affinity channel as last channel
    assert affinity_maps.ndim == 4
    assert affinity_maps.shape[-1] == lifted_range

    rag = ds.rag(seg_id)
    adjacency = get_long_range_z_adjacency(ds, seg_id, lifted_range)

    if ds.has_seg_mask:
        long_range_feats = nrag.accumulateLongRangeFeatures(
            rag,
            affinity_maps,
            len(adjacency),
            ignoreSegValue=ExperimentSettings().ignore_seg_value,
            numberOfThreads=ExperimentSettings().n_threads
        )
    else:
        long_range_feats = nrag.accumulateLongRangeFeatures(
            rag,
            affinity_maps,
            len(adjacency),
            numberOfThreads=ExperimentSettings().n_threads
        )
    return np.nan_to_num(long_range_feats)


# match the standard lifted neighborhood (lifted-nh) to the z-adjacency (lifted-range)
def match_to_lifted_nh(ds, seg_id, lifted_nh, lifted_range):
    assert lifted_nh >= lifted_range
    uv_lifted = compute_and_save_lifted_nh(ds, seg_id, lifted_nh)
    uv_long_range = get_long_range_z_adjacency(ds, seg_id, lifted_range)
    # match
    matches = find_matching_row_indices(uv_long_range, uv_lifted)[:, 0]
    assert len(matches) == len(uv_long_range)
    return matches


# learn lifted rf with long ange features
def learn_rf_with_long_range():
    pass
