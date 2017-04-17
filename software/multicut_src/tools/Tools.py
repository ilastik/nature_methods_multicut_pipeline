import vigra
import os
import numpy as np

from functools import wraps
from itertools import combinations, product

def cache_name(fname, folder_str, ignoreNp, edge_feat_cache, *args):
    self = args[0]
    arg_id = 1
    for arg in args[1:]:
        # for the edgefeats we have to clip the anisotropy
        # factor if it is larger than max. aniso factor
        if edge_feat_cache and arg_id == 3:
            if arg >= self.aniso_max:
                arg = self.aniso_max
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
def cacher_hdf5(folder = "dset_folder", cache_edgefeats = False, ignoreNumpyArrays=False):
    assert folder in ("dset_folder", "feature_folder")
    _folder = folder
    _cache_edgefeats = cache_edgefeats
    def actualDecorator(function):
        @wraps(function)
        # for now, we dont support keyword arguments!
        def wrapper(*args):
            fname = str(function.__name__)
            self = args[0]
            filepath = cache_name(fname, _folder, ignoreNumpyArrays, _cache_edgefeats, *args)
            fkey  = "data"
            if not os.path.isfile(filepath):
                print "Computing: ", function.__name__, "with args:"
                print args[1:]
                print "Results will be written in ", filepath, fkey
                _res = function(*args)
                vigra.writeHDF5(_res, filepath, fkey, compression = self.compression)
            else:
                _res = vigra.readHDF5(filepath, fkey)
            return _res
        return wrapper
    return actualDecorator


# for visualizing edges
def edges_to_volume(rag, edges, ignore_z = False):

    assert rag.edgeNum == edges.shape[0], str(rag.edgeNum) + " , " + str(edges.shape[0])

    print rag.baseGraph.shape
    volume = np.zeros(rag.baseGraph.shape, dtype = np.uint32)

    for edge_id in rag.edgeIds():
        # don't write the ignore label!
        if edges[edge_id] == 0:
            continue
        edge_coords = rag.edgeCoordinates(edge_id)
        if ignore_z:
            if edge_coords[0,2] - int(edge_coords[0,2]) != 0:
                continue

        edge_coords_up = ( np.ceil(edge_coords[:,0]).astype(np.uint32),
                np.ceil(edge_coords[:,1]).astype(np.uint32),
                np.ceil(edge_coords[:,2]).astype(np.uint32) )
        edge_coords_dn = ( np.ceil(edge_coords[:,0]).astype(np.uint32),
                np.ceil(edge_coords[:,1]).astype(np.uint32),
                np.ceil(edge_coords[:,2]).astype(np.uint32) )

        volume[edge_coords_dn] = edges[edge_id]
        volume[edge_coords_up] = edges[edge_id]

    return volume


# for visualizing in plane edges
@cacher_hdf5(ignoreNumpyArrays=True)
def edges_to_volume_from_uvs_in_plane(ds, seg, uv_ids, edge_labels):
    assert uv_ids.shape[0] == edge_labels.shape[0]
    from cython_tools import fast_edge_volume_from_uvs_in_plane
    print "Computing edge volume from uv ids in plane"
    return fast_edge_volume_from_uvs_in_plane(seg, uv_ids, edge_labels)


# for visualizing between edges
@cacher_hdf5(ignoreNumpyArrays=True)
def edges_to_volume_from_uvs_between_plane(ds, seg, uv_ids, edge_labels, look_dn):
    assert uv_ids.shape[0] == edge_labels.shape[0]
    from cython_tools import fast_edge_volume_from_uvs_between_plane
    print "Computing edge volume from uv ids between planes"
    return fast_edge_volume_from_uvs_between_plane(seg, uv_ids, edge_labels, look_dn)


# for visualizing skip edges
@cacher_hdf5(ignoreNumpyArrays=True)
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
    volume = np.zeros(seg.shape, dtype = edge_labels.dtype)

    # find all the slices with defect starts
    lower_slices  = np.unique(skip_starts)
    skip_masks_to_lower = {z : skip_starts == z for z in lower_slices}

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

            seg_dn = seg[:,:,lower]
            seg_up = seg[:,:,upper]

            # get the mask for skip edges connecting to this upper slice
            mask_upper = ranges_lower == unique_ranges[i]
            uvs_to_upper = np.sort(uvs_lower[mask_upper], axis = 1)
            assert uvs_to_upper.shape[1] == 2
            labels_upper = labels_lower[mask_upper]

            # for debugging
            #uniques_up = np.unique(seg_up)
            #uniques_dn = np.unique(seg_dn)
            #unique_uvs = np.unique(uvs_to_upper)
            ## this should more or less add up (except for bg value)
            #matches_dn = np.intersect1d(uniques_up, unique_uvs).size
            #matches_up = np.intersect1d(uniques_dn, unique_uvs).size
            #print "Matches_up", matches_up, '/', unique_uvs.size
            #print "Matches_dn", matches_dn, '/', unique_uvs.size
            #print "Combined:", matches_up + matches_dn, '/', unique_uvs.size
            #assert seg_dn.shape == seg_up.shape, "%s, %s" % (str(seg_dn.shape), str(seg_up.shape))

            vol_dn, vol_up = fast_edge_volume_for_skip_edges_slice(
                    seg_dn,
                    seg_up,
                    uvs_to_upper,
                    labels_upper)
            volume[:,:,lower] = vol_dn
            volume[:,:,upper] = vol_up

    return volume
