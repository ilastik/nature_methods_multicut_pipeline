import numpy as np
import vigra

from ExperimentSettings import ExperimentSettings

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
# TODO test this properly!
def merge_small_segments(mc_seg, min_seg_size):

    # take care of segmentations that don't start at zero
    seg_min = mc_seg.min()
    if seg_min > 0:
        seg -= seg_min

    n_threads = ExperimentSettings().n_threads
    rag = nifty.graph.rag.gridRag(mc_seg, n_threads)
    n_nodes = rag.numberOfNodes
    assert n_nodes == mc_seg.max() + 1, "%i, %i" % (n_nodes, mc_seg.max()+1)

    print "Merging segments in mc-result with size smaller than", min_seg_size
    _, node_sizes = np.unique(mc_seg, return_counts = True)
    edge_sizes = nifty.graph.rag.accumulateEdgeMeanAndLength(
            rag,
            np.zeros_like(mc_seg, dtype = 'float32')
    )[:,1].astype('uint32')
    assert len(node_sizes) == n_nodes

    # find nodes that shall be merged
    merge_nodes = np.where(node_sizes < min_seg_size)[0]

    # iterate over the merge nodes and merge with adjacent
    # node with biggest overlap
    merge_pairs = []
    for u in merge_nodes:
        merge_n_id    = -1
        max_edge_size = 0
        for adj in rag.nodeAdjacency(u):
            v, edge_id = adj[0], adj[1]
            edge_size = edge_sizes[edge_id]
            if edge_size > max_edge_size:
                max_edge_size = edge_size
                merge_n_id = v
        assert merge_n_id != -1
        merge_pairs.append( [u,merge_n_id] )
    merge_pairs = np.array(merge_pairs)

    # merge the nodes with ufd
    ufd = nifty.ufd.ufd(n_nodes)
    ufd.merge(merge_pairs)
    merged_nodes = ufd.elmentLabeling()

    # make consecutive, starting from the original min val and make segmentation
    merged_nodes,_,_ = vigra.analysis.relabelConsecutive(merged_nodes, start_label = seg_min, keep_zeros = False)
    return nifty.graph.rag.projectScalarNodeDataToPixels(rag, merged_nodes, n_threads)
