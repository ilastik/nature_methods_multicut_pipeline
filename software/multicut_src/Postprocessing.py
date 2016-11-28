import numpy as np
import vigra

from Tools import UnionFind

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
