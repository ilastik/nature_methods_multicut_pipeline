import vigra
import os
import numpy as np

from functools import wraps

from multicut_src.ExperimentSettings import ExperimentSettings

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex as nifty  # conda version build with cplex
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi as nifty  # conda version build with gurobi
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


def cache_name(fname, folder_str, ignoreNp, edge_feat_cache, *args):
    self = args[0]
    arg_id = 1
    for arg in args[1:]:
        # for the edgefeats we have to clip the anisotropy
        # factor if it is larger than max. aniso factor
        if edge_feat_cache and arg_id == 3:
            if arg >= ExperimentSettings().aniso_max:
                arg = ExperimentSettings().aniso_max
        if isinstance(arg, np.ndarray) and not ignoreNp:
            fname += "_" + str(arg)
        elif isinstance(arg, np.ndarray):
            pass
        # need to make tuples and lists cacheable
        elif isinstance(arg, list) or isinstance(arg, tuple):
            for elem in arg:
                fname += "_" + str(elem) + "_"
        else:
            fname += "_" + str(arg)
        arg_id += 1
    fname += ".h5"
    if folder_str == "dset_folder":
        save_folder = self.cache_folder
    elif folder_str == "feature_folder":
        save_folder = os.path.join(self.cache_folder, "features")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    return os.path.join(save_folder, fname)


# TODO check for arguments too long for caching
# cache result as hdf5
def cacher_hdf5(folder="dset_folder", cache_edgefeats=False, ignoreNumpyArrays=False, compress=False):
    assert folder in ("dset_folder", "feature_folder")
    _folder = folder
    _cache_edgefeats = cache_edgefeats

    def actualDecorator(function):
        @wraps(function)
        # for now, we dont support keyword arguments!
        def wrapper(*args):
            fname = str(function.__name__)
            filepath = cache_name(fname, _folder, ignoreNumpyArrays, _cache_edgefeats, *args)
            fkey  = "data"
            if not os.path.isfile(filepath):
                print "Computing: ", function.__name__, "with args:"
                print args[1:]
                print "Results will be written in ", filepath, fkey
                _res = function(*args)
                if compress:
                    vigra.writeHDF5(_res, filepath, fkey, compression='gzip')
                else:
                    vigra.writeHDF5(_res, filepath, fkey)
                # compressing does not make much sense for most of the files we cache
                # vigra.writeHDF5(_res, filepath, fkey, compression = ExperimentSettings().compression)
            else:
                _res = vigra.readHDF5(filepath, fkey)
            return _res
        return wrapper
    return actualDecorator


# FIXME this is deprecated, use nrag.ragCoordinates.edgesToVolume
# for visualizing edges
def edges_to_volume(rag, edges, edge_direction=0):

    assert rag.numberOfEdges == edges.shape[0], str(rag.numberOfEdges) + " , " + str(edges.shape[0])

    rag_coords = nrag.ragCoordinates(rag, numberOfThreads=ExperimentSettings().n_threads)
    return rag_coords.edgesToVolume(
        edges,
        edgeDirection=edge_direction,
        numberOfThreads=ExperimentSettings().n_threads
    )


# for visualizing in plane edges
@cacher_hdf5(ignoreNumpyArrays=True, compress=True)
def edges_to_volume_from_uvs_in_plane(ds, seg, uv_ids, edge_labels):
    assert uv_ids.shape[0] == edge_labels.shape[0]
    from cython_tools import fast_edge_volume_from_uvs_in_plane
    print "Computing edge volume from uv ids in plane"
    return fast_edge_volume_from_uvs_in_plane(seg, uv_ids, edge_labels.astype('uint8'))


# for visualizing between edges
@cacher_hdf5(ignoreNumpyArrays=True, compress=True)
def edges_to_volume_from_uvs_between_plane(ds, seg, uv_ids, edge_labels):
    assert uv_ids.shape[0] == edge_labels.shape[0]
    from cython_tools import fast_edge_volume_from_uvs_between_plane
    print "Computing edge volume from uv ids between planes"
    return fast_edge_volume_from_uvs_between_plane(seg, uv_ids, edge_labels.astype('uint8'))


# for visualizing skip edges
@cacher_hdf5(ignoreNumpyArrays=True, compress=True)
def edges_to_volumes_for_skip_edges(
        ds,
        seg,
        uv_ids,
        edge_labels,
        skip_starts,
        skip_ranges):

    assert uv_ids.shape[0] == edge_labels.shape[0]
    assert skip_ranges.shape[0] == uv_ids.shape[0]
    assert skip_starts.shape[0] == uv_ids.shape[0]
    assert uv_ids.shape[1] == 2

    from cython_tools import fast_edge_volume_for_skip_edges_slice
    print "Computing edge volume for skip edges"
    volume = np.zeros(seg.shape, dtype=edge_labels.dtype)

    # find all the slices with defect starts
    lower_slices  = np.unique(skip_starts)
    skip_masks_to_lower = {z: skip_starts == z for z in lower_slices}

    # iterate over the slice pairs with skip edges and get the label volumes from cython
    for lower in lower_slices:
        print "Slice", lower
        # get the uvs and ranges for this lower slice
        mask_lower = skip_masks_to_lower[lower]
        ranges_lower = skip_ranges[mask_lower]
        labels_lower = edge_labels[mask_lower]
        uvs_lower    = uv_ids[mask_lower]
        # get the target slcies from unique ranges
        unique_ranges = np.unique(ranges_lower)
        targets = unique_ranges + lower
        for i, upper in enumerate(targets):
            print "to", upper

            seg_dn = seg[lower]
            seg_up = seg[upper]
            print seg_dn.shape
            print seg_up.shape
            assert seg_dn.shape == seg_up.shape, "%s, %s" % (str(seg_dn.shape), str(seg_up.shape))

            # get the mask for skip edges connecting to this upper slice
            mask_upper = ranges_lower == unique_ranges[i]
            uvs_to_upper = np.sort(uvs_lower[mask_upper], axis=1)
            assert uvs_to_upper.shape[1] == 2
            labels_upper = labels_lower[mask_upper]

            vol_dn, vol_up = fast_edge_volume_for_skip_edges_slice(
                seg_dn,
                seg_up,
                uvs_to_upper,
                labels_upper.astype('uint8')
            )
            volume[lower] = vol_dn
            volume[upper] = vol_up

    return volume
