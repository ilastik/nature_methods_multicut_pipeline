import numpy as np
import vigra
from concurrent import futures

# if build from sorce and not a conda pkg, we assume that we have cplex
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty # conda version build with cplex
    except ImportError:
        try:
            import nifty_wit_gurobi as nifty # conda version build with gurobi
        except ImportError:
            raise ImportError("No valid nifty version was found.")

from tools import replace_from_dict


# TODO 10,000 seems to be a pretty large default value !
# TODO rethink the relabeling here, in which cases do we want it, can it hurt?
def remove_small_segments(segmentation,
        size_thresh = 10000,
        relabel = True,
        return_sizes = False):

    # Make sure all objects have their individual label
    # NOTE this is very dangerous for sample C (black slices in groundtruth)
    if relabel:
        segmentation = vigra.analysis.labelMultiArrayWithBackground(
            segmentation.astype('uint32'),
            background_value = 0,
            neighborhood = 'indirect')

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

    if return_sizes:
        return segmentation, counts[counts >= size_thresh]
    else:
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

    assert n_nodes == mc_seg.max(), str(n_nodes) + " , " + str(mc_seg.max())
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
    ufd = nifty.ufd.ufd( n_nodes + 1 )
    for merge_pair in merge_nodes:
        ufd.merge(merge_pair[0], merge_pair[1])

    # get new to old as merge result
    merged_nodes = ufd.elmentLabeling()

    # merge the new nodes
    merged_seg = seg_rag.projectLabelsToBaseGraph(merged_nodes)
    merged_seg = vigra.analysis.labelVolume(merged_seg)
    return merged_seg
