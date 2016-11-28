import vigra
import os
import cPickle as pickle
import numpy as np

from functools import wraps
from itertools import combinations

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


# TODO check for arguments too long for caching
# TODO log instead of printing, because all these prints become super annoying
# cache result as hdf5
def cacher_hdf5(folder = "dset_folder", cache_edgefeats = False,ignoreNumpyArrays=False):
    assert folder in ("dset_folder", "feature_folder")
    _folder = folder
    _cache_edgefeats = cache_edgefeats
    def actualDecorator(function):
        @wraps(function)
        # for now, we dont support keyword arguments!
        def wrapper(*args):

            self = args[0]

            fname = str(function.__name__)
            arg_id = 1
            for arg in args[1:]:
                # for the edgefeats we have to clip the anisotropy
                # factor if it is larger than max. aniso factor
                if _cache_edgefeats and arg_id == 3:
                    if arg >= self.aniso_max:
                        arg = self.aniso_max
                if isinstance(arg, np.ndarray) and not ignoreNumpyArrays:
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
            if _folder == "dset_folder":
                save_folder = self.cache_folder
            elif _folder == "feature_folder":
                save_folder = os.path.join(self.cache_folder, "features")
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            filepath = os.path.join(save_folder, fname)
            fkey  = "data"

            if not os.path.isfile(filepath):
                print "Computing: ", function.__name__, "with args:"
                print args[1:]
                print "Results will be written in ", filepath, fkey
                _res = function(*args)
                #print "Writing result in ", filepath, fkey
                vigra.writeHDF5(_res, filepath, fkey, compression = self.compression)

            else:
                #print "Loading: ", function.__name__, "with args:"
                #print args[1:]
                #print "From:"
                #print filepath
                #print fkey
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
    volume = np.zeros(rag.baseGraph.shape, dtype = np.uint8)

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


def edges_to_binary(rag, edges, project_2d = True):

    assert rag.edgeNum == edges.shape[0], str(rag.edgeNum) + " , " + str(edges.shape[0])

    binary = 255 * np.ones(rag.baseGraph.shape, dtype = np.uint8)

    for edge_id in rag.edgeIds():
        if edges[edge_id]:
            edge_coords = rag.edgeCoordinates(edge_id)
            # if project 2d is set, we only project the xy edges
            if project_2d:
                z = edge_coords[:,2]
                assert np.unique(z).shape[0] == 1
                z = z[0]
                if z - int(z) != 0:
                    continue

            edge_coords_up = ( np.ceil(edge_coords[:,0]).astype(np.uint32),
                    np.ceil(edge_coords[:,1]).astype(np.uint32),
                    np.ceil(edge_coords[:,2]).astype(np.uint32) )
            edge_coords_dn = ( np.ceil(edge_coords[:,0]).astype(np.uint32),
                    np.ceil(edge_coords[:,1]).astype(np.uint32),
                    np.ceil(edge_coords[:,2]).astype(np.uint32) )

            binary[edge_coords_dn] = 0
            binary[edge_coords_up] = 0

    return binary


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
