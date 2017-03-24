import numpy as np
import vigra
from concurrent import futures

from Tools import UnionFind

# numpy.replace: replcaces the values in array according to dict
# cf. SO: http://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
def replace_from_dict(array, dict_like):
    replace_keys, replace_vals = np.array(list(zip( *sorted(dict_like.items() ))))
    # FIXME This is just some dirty hack because I can't get np version 1.10 to run
    if np.__version__ == '1.9.3':
        indices = np.digitize(array.flatten(), replace_keys, right=True)
        return replace_vals[indices].astype(array.dtype).reshape(array.shape)
    else:
        indices = np.digitize(array, replace_keys, right = True)
        return replace_vals[indices].astype(array.dtype)

# TODO 10,000 seems to be a pretty large default value !
# TODO FIXME rethink the relabeling here, in which cases do we want it, can it hurt?
def remove_small_segments(segmentation,
        size_thresh = 10000,
        relabel = True):

    # Make sure all objects have their individual label
    # TODO FIXME this is very dangerous for sample C (black slices !)!
    if relabel:
        segmentation = vigra.analysis.labelVolumeWithBackground(
            segmentation.astype('uint32'), neighborhood=6, background_value=0)

    # Get the unique values of the segmentation including counts
    uniq, counts = np.unique(segmentation, return_counts=True)

    # Keep all uniques that have a count smaller than size_thresh
    small_objs = uniq[counts < size_thresh]
    large_objs = uniq[counts >= size_thresh]
    print 'len(large_objs) == {}'.format(len(large_objs))
    print 'len(small_objs) == {}'.format(len(small_objs))

    # I think this is the fastest (single threaded way) to do this
    # If we really need to parallelize this, we need to rethink a little, but for now, this should be totally fine!
    if relabel:
        large_objs_to_consecutive = {obj_id : i+1 for i, obj_id in enumerate(large_objs)}
        obj_dict = {obj_id : 0 if obj_id in small_objs else large_objs_to_consecutive[obj_id] for obj_id in uniq}
    else:
        obj_dict = {obj_id : 0 if obj_id in small_objs else obj_id for obj_id in uniq}
    segmentation = replace_from_dict(segmentation, obj_dict)
    return segmentation


# merge segments that are smaller than min_seg_size
# TODO more efficiently ?!
# TODO test this properly!
def merge_small_segments(mc_seg, min_seg_size):
    seg_rag = vigra.graphs.regionAdjacencyGraph(
            vigra.graphs.gridGraph(mc_seg.shape[0:3]),
            mc_seg.astype(np.uint32) )

    # zero should be reserved for the ignore label!
    assert 0 not in mc_seg

    n_nodes = seg_rag.nodeNum

    # FIXME This caused a yet not investigated error
    # assert n_nodes == mc_seg.max(), str(n_nodes) + " , " + str(mc_seg.max())
    if not n_nodes == mc_seg.max():
        print "Warning: (n_nodes = {}) != (mc_seg.max() = {})".format(n_nodes, mc_seg.max())

    print "Merging segments in mc-result with size smaller than", min_seg_size

    seg_sizes = np.bincount(mc_seg.ravel())

    segs_merge = np.zeros(n_nodes+1, dtype = bool)
    segs_merge[seg_sizes <= min_seg_size] = True
    print "Merging", np.sum(segs_merge), "segments"

    merge_nodes = []
    for node in seg_rag.nodeIter():
        n_id = node.id
        # if the node id is not zero and it is marked in segs_merge, we merge it
        if n_id != 0 and segs_merge[n_id]:
            # need to find the adjacent node with largest edge
            max_edge_size = 0
            merge_node_id = -1
            for adj_node in seg_rag.neighbourNodeIter(node):
                edge = seg_rag.findEdge(node,adj_node)
                edge_size = len( seg_rag.edgeCoordinates(edge) )
                if edge_size > max_edge_size:
                    max_edge_size = edge_size
                    merge_node_id = adj_node.id
            assert merge_node_id != -1
            merge_nodes.append( (n_id, merge_node_id) )

    # merge the nodes with udf
    udf = UnionFind( n_nodes + 1 )
    for merge_pair in merge_nodes:
        udf.merge(merge_pair[0], merge_pair[1])

    # get new to old as merge result
    new_to_old = udf.get_merge_result()

    # find old to new nodes
    old_to_new = np.zeros( n_nodes + 1, dtype = np.uint32 )
    for set_id in xrange( len(new_to_old)  ):
        for n_id in new_to_old[set_id]:
            assert n_id <= n_nodes, str(n_id) + " , " + str(n_nodes)
            old_to_new[n_id] = set_id

    # merge the new nodes
    merged_seg = seg_rag.projectLabelsToBaseGraph(old_to_new)
    merged_seg = vigra.analysis.labelVolume(merged_seg)
    return merged_seg
