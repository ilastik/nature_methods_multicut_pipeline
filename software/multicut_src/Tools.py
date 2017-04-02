import vigra
import os
import cPickle as pickle
import numpy as np

from functools import wraps
from itertools import combinations, product
from concurrent import futures

#
# Implementation of a disjoint-set forest
#

# TODO maybe use C++ implementation instead
# Node datastrucuture for UDF
# works only for connected labels
class Node(object):
    def __init__(self, u):
        self.parent = self
        self.label  = u
        self.rank   = 0

class UnionFind(object):

    def __init__(self, n_labels):
        assert isinstance(n_labels, int), type(n_labels)
        self.n_labels = n_labels
        self.nodes = [Node(n) for n in xrange(n_labels)]


    # find the root of u and compress the path on the way
    def find(self, u_id):
        #assert u_id < self.n_labels
        u = self.nodes[ u_id ]
        return self.findNode(u)

    # find the root of u and compress the path on the way
    def findNode(self, u):
        if u.parent == u:
            return u
        else:
            u.parent = self.findNode(u.parent)
            return u.parent

    def merge(self, u_id, v_id):
        #assert u_id < self.n_labels
        #assert v_id < self.n_labels
        u = self.nodes[ u_id ]
        v = self.nodes[ v_id ]
        self.mergeNode(u, v)

    # merge u and v trees in a union by rank manner
    def mergeNode(self, u, v):
        u_root = self.findNode(u)
        v_root = self.findNode(v)
        if u_root.rank > v_root.rank:
            v_root.parent = u_root
        elif u_root.rank < v_root.rank:
            u_root.parent = v_root
        elif u_root != v_root:
            v_root.parent = u_root
            u_root.rank += 1

    # get the new sets after merging
    def get_merge_result(self):

        merge_result = []

        # find all the unique roots
        roots = []
        for u in self.nodes:
            root = self.findNode(u)
            if not root in roots:
                roots.append(root)

        # find ordering of roots (from 1 to n_roots)
        roots_ordered = {}
        root_id = 0
        for root in roots:
            merge_result.append( [] )
            roots_ordered[root] = root_id
            root_id += 1
        for u in self.nodes:
            u_label = u.label
            root = self.findNode(u)
            merge_result[ roots_ordered[root] ].append(u_label)

        # sort the nodes in the result
        #(this might result in problems if label_type cannot be sorted)
        for res in merge_result:
            res.sort()

        return merge_result

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
def edges_to_volume_from_uvs_between_plane(ds, seg, uv_ids, edge_labels):
    assert uv_ids.shape[0] == edge_labels.shape[0]
    from cython_tools import fast_edge_volume_from_uvs_between_plane
    print "Computing edge volume from uv ids between planes"
    return fast_edge_volume_from_uvs_between_plane(seg, uv_ids, edge_labels)


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
    from cython_tools import fast_edge_volume_for_skip_edges_slice
    print "Computing edge volume for skip edges"
    volume = np.zeros(seg.shape, dtype = edge_labels.dtype)

    # find all the slices with defect starts
    lower_slices  = np.unique(skip_starts)
    skip_edge_indices_to_slice = {z : np.where(skip_starts == z)[0] for z in lower_slices}
    target_slices = {z : z + np.unique(skip_ranges[skip_starts == z]) for z in lower_slices}
    # this only works for a single target slice for now
    for z in target_slices:
        assert target_slices[z].size == 1, str(z)
        target_slices[z] = target_slices[z][0]

    # iterate over the slice pairs with skip edges and get the label volumes from cython
    for lower in lower_slices:
        upper = target_slices[lower]
        #print "From", lower, "to", upper
        seg_dn = seg[:,:,lower]
        seg_up = seg[:,:,upper]
        assert seg_dn.shape == seg_up.shape, "%s, %s" % (str(seg_dn.shape), str(seg_up.shape))
        vol_dn, vol_up = fast_edge_volume_for_skip_edges_slice(
                seg_dn,
                seg_up,
                uv_ids[skip_edge_indices_to_slice[lower]],
                edge_labels[skip_edge_indices_to_slice[lower]])
        volume[:,:,lower] = vol_dn
        volume[:,:,upper] = vol_up

    return volume

    ##for e_id, uv in enumerate(uv_ids):
    #def _write_coords(e_id, uv):
    #    #print e_id, '/', n_edges
    #    val = edge_labels[e_id]
    #    if val == 0: # 0 labels are ignored
    #        return False
    #    u, v = uv
    #    coords_u = np.where(seg == u)
    #    coords_v = np.where(seg == v)
    #    coords_u = np.concatenate(
    #            [coords_u[0][:,None], coords_u[1][:,None], coords_u[2][:,None]],
    #            axis = 1 )
    #    coords_v = np.concatenate(
    #            [coords_v[0][:,None], coords_v[1][:,None], coords_v[2][:,None]],
    #            axis = 1 )
    #    z_u = np.unique(coords_u[:,2])
    #    z_v = np.unique(coords_v[:,2])
    #    assert z_u.size == 1
    #    assert z_v.size == 1
    #    z_u, z_v = z_u[0], z_v[0]
    #    assert z_u != z_v, "%i, %i" % (z_u, z_v)

    #    # for z-edges find the intersection in plane
    #    # get the intersecting coordinates:
    #    # cf: http://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    #    intersect = np.array([x for x in ( set(tuple(x) for x in coords_u[:,:2]) & set(tuple(x) for x in coords_v[:,:2]) ) ])
    #    intersect_u = (
    #            intersect[:,0],
    #            intersect[:,1],
    #            z_u * np.ones(intersect.shape[0], intersect.dtype) )
    #    intersect_v = (
    #            intersect[:,0],
    #            intersect[:,1],
    #            z_v * np.ones(intersect.shape[0], intersect.dtype) )
    #    volume[intersect_u] = val
    #    volume[intersect_v] = val

    ## serial for debugging
    ##res = [ _write_coords(e_id, uv) for e_id, uv in enumerate(uv_ids) ]

    ## parallel
    #with futures.ThreadPoolExecutor(max_workers = 8) as executor:
    #    tasks = [ executor.submit(_write_coords, e_id, uv) for e_id, uv in enumerate(uv_ids) ]
    #    res = [t.result() for t in tasks]

    #return volume


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
