import vigra
import os
import cPickle as pickle
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
# TODO log instead of printing, because all these prints become super annoying
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


# TODO check for arguments too long for caching
# TODO this is ridiculously slow and takes a huge amount of space!
# cache result as pickle
def cacher_pickle():
    def actualDecorator(function):
        @wraps(function)
        # for now, we dont support keyword arguments!
        def wrapper(*args):

            self = args[0]

            fname = str(function.__name__)
            for arg in args[1:]:
                fname += "_" + str(arg)
            fname += ".pkl"
            filepath = os.path.join(self.cache_folder, fname)

            if not os.path.isfile(filepath):
                print "Computing: ", function.__name__, "with args:"
                print args[1:]
                _res = function(*args)

                print "Writing result in ", filepath
                with open(filepath, 'w') as f:
                    pickle.dump(_res, f)

            else:
                print "Loading: ", function.__name__, "with args:"
                print args[1:]
                print "From:"
                print filepath
                print rag_key
                with open(filepath, 'r') as f:
                    _res = pickle.load(f)

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

# DEPRECATED, in blockwise-mc replace with vigra or nifty blocking

# find the coordinates of all blocks for a tesselation of
# vol_shape with block_shape and n_blocks
def find_block_coordinates(vol_shape, block_shape, n_blocks, is_covering):
    assert n_blocks in (2,4,8)
    # init block coordinates with 0 lists for all the cutouts
    block_coordinates = []
    for block_id in range(n_blocks):
        block_coordinates.append( [0,0,0,0,0,0] )

    # 2: block 0 = lower coordinate in non-spanning coordinate
    if n_blocks == 2:
        assert np.sum(is_covering) == 2, "Need exacly two dimensions fully covered to cover the volume with 2 blocks."
        split_coord =  np.where( np.array( is_covering )  == False )[0][0]

        for dim in range(3):
            index_dn = 2*dim
            index_up = 2*dim + 1
            if dim == split_coord:
                block_coordinates[0][index_dn] = 0
                block_coordinates[0][index_up] = block_shape[dim]
                block_coordinates[1][index_dn] = vol_shape[dim] - block_shape[dim]
                block_coordinates[1][index_up] = vol_shape[dim]
            else:
                assert block_shape[dim] == vol_shape[dim]
                block_coordinates[0][index_dn] = 0
                block_coordinates[0][index_up] = vol_shape[dim]
                block_coordinates[1][index_dn] = 0
                block_coordinates[1][index_up] = vol_shape[dim]

    # 4 : block 0 -> 0:split1,0:split2
    # block 1 -> split1:,0:split2
    # block 2 -> 0:split1,split2:
    # block 3 -> split1:,split2:
    if n_blocks == 4:
        assert np.sum(is_covering) == 1, "One dimension must be fully covered by a single block for 4 subblocks."
        split_coords = np.where( np.array( is_covering )  == False )[0]
        for dim in range(3):

            index_dn = 2*dim
            index_up = 2*dim + 1

            if dim in split_coords:
                block_coordinates[0][index_dn] = 0
                block_coordinates[0][index_up] = block_shape[dim]

                block_coordinates[3][index_dn] = vol_shape[dim] - block_shape[dim]
                block_coordinates[3][index_up] = vol_shape[dim]

                if dim == split_coords[0]:
                    block_coordinates[1][index_dn] = vol_shape[dim] - block_shape[dim]
                    block_coordinates[1][index_up] = vol_shape[dim]

                    block_coordinates[2][index_dn] = 0
                    block_coordinates[2][index_up] = block_shape[dim]

                else:
                    block_coordinates[1][index_dn] = 0
                    block_coordinates[1][index_up] = block_shape[dim]

                    block_coordinates[2][index_dn] = vol_shape[dim] - block_shape[dim]
                    block_coordinates[2][index_up] = vol_shape[dim]
            else:
                for block_id in range(n_blocks):
                    assert block_shape[dim] == vol_shape[dim]
                    block_coordinates[block_id][index_dn] = 0
                    block_coordinates[block_id][index_up] = block_shape[dim]

    # 8 : block 0 -> 0:split1,0:split2,0:split3
    # block 1 -> split1: ,0:split2,0:split3
    # block 2 -> 0:split1,split2: , 0:split3
    # block 3 -> 0:split1,0:split2, split3:
    # block 4 -> split1: ,split2: ,0:split3
    # block 5 -> split1: ,0:split2,split3:
    # block 6 -> 0:split1,split2: ,split3:
    # block 7 -> split1:,split2:,split3:
    if n_blocks == 8:
        assert np.sum(is_covering) == 0, "No dimension can be fully covered by a single block for 8 subblocks."
        # calculate the split coordinates
        x_split = vol_shape[0] - block_shape[0]
        y_split = vol_shape[1] - block_shape[1]
        z_split = vol_shape[2] - block_shape[2]
        # all coords are split coords, this makes life a little easier
        block_coordinates[0] = [0,block_shape[0],
                                0,block_shape[1],
                                0,block_shape[2]]
        block_coordinates[1] = [x_split, vol_shape[0],
                                0,block_shape[1],
                                0,block_shape[2]]
        block_coordinates[2] = [0, block_shape[0],
                                y_split,vol_shape[1],
                                0,block_shape[2]]
        block_coordinates[3] = [0, block_shape[0],
                                0, block_shape[1],
                                z_split,vol_shape[2]]
        block_coordinates[4] = [x_split, vol_shape[0],
                                y_split, vol_shape[1],
                                0, block_shape[2]]
        block_coordinates[5] = [x_split, vol_shape[0],
                                0, block_shape[1],
                                z_split, vol_shape[2]]
        block_coordinates[6] = [0, block_shape[0],
                               y_split, vol_shape[1],
                               z_split, vol_shape[2]]
        block_coordinates[7] = [x_split, vol_shape[0],
                                y_split, vol_shape[1],
                                z_split, vol_shape[2]]

    return block_coordinates


# find the coordinate overlaps between block_coordinates
# for the given pairs of blocks and the vol_shape
def find_overlaps(pairs, vol_shape, block_coordinates, is_covering):
    #assert len(block_coordinates) - 1 == np.max(pairs), str(len(block_coordinates) - 1) + " , " + str(np.max(pairs))
    overlaps = {}
    for pair in pairs:
        ovlp = [0,0,0,0,0,0]
        for dim in range(3):
            index_dn = 2*dim
            index_up = 2*dim + 1
            if is_covering[dim]:
                # if the dimension is fully covered, the overlap is over the whole dimension
                ovlp[index_dn] = 0
                ovlp[index_up] = vol_shape[dim]
            else:
                # if the dimension is not fully covered, we have to find the overlap
                # either the pair has the same span in this dimension
                # (than the whole dim is overlapping))
                # or we have a real overlap and have to get this!
                dn_0 = block_coordinates[pair[0]][index_dn]
                dn_1 = block_coordinates[pair[1]][index_dn]
                up_0 = block_coordinates[pair[0]][index_up]
                up_1 = block_coordinates[pair[1]][index_up]
                if dn_0 == dn_1:
                    assert up_0 == up_1, "DIM:" + str(dim) + " , " + str(up_0) + " , " + str(up_1)
                    ovlp[index_dn] = dn_0
                    ovlp[index_up] = up_0
                else:
                    min_pair = pair[ np.argmin( [dn_0, dn_1] ) ]
                    max_pair = pair[ np.argmax( [dn_0, dn_1] ) ]
                    assert min_pair != max_pair
                    ovlp[index_dn] = block_coordinates[max_pair][index_dn]
                    ovlp[index_up] = block_coordinates[min_pair][index_up]

        overlaps[pair] = ovlp
    return overlaps


# get all blockpairs
def get_block_pairs(n_blocks):
    return combinations( range(n_blocks), 2 )
