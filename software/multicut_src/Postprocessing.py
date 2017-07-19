import numpy as np
import vigra
from concurrent import futures

from ExperimentSettings import ExperimentSettings

# if build from sorce and not a conda pkg, we assume that we have cplex
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

from tools import replace_from_dict


# TODO 10,000 seems to be a pretty large default value !
# TODO rethink the relabeling here, in which cases do we want it, can it hurt?
def remove_small_segments(
    segmentation,
    size_thresh=10000,
    relabel=True,
    return_sizes=False
):

    # Make sure all objects have their individual label
    # NOTE this is very dangerous for sample C (black slices in groundtruth)
    if relabel:
        segmentation = vigra.analysis.labelMultiArrayWithBackground(
            segmentation.astype('uint32'),
            background_value=0,
            neighborhood='indirect'
        )

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
        large_objs_to_consecutive = {obj_id: i + 1 for i, obj_id in enumerate(large_objs)}
        obj_dict = {obj_id: 0 if obj_id in small_objs else large_objs_to_consecutive[obj_id] for obj_id in uniq}
    else:
        obj_dict = {obj_id: 0 if obj_id in small_objs else obj_id for obj_id in uniq}
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
        mc_seg -= seg_min

    n_threads = ExperimentSettings().n_threads
    rag = nrag.gridRag(mc_seg, n_threads)
    n_nodes = rag.numberOfNodes
    assert n_nodes == mc_seg.max() + 1, "%i, %i" % (n_nodes, mc_seg.max() + 1)

    print "Merging segments in mc-result with size smaller than", min_seg_size
    _, node_sizes = np.unique(mc_seg, return_counts=True)
    edge_sizes = nrag.accumulateEdgeMeanAndLength(
        rag,
        np.zeros_like(mc_seg, dtype='float32')
    )[:, 1].astype('uint32')
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
        merge_pairs.append([u, merge_n_id])
    merge_pairs = np.array(merge_pairs)

    # merge the nodes with ufd
    ufd = nifty.ufd.ufd(n_nodes)
    ufd.merge(merge_pairs)
    merged_nodes = ufd.elementLabeling()

    # make consecutive, starting from the original min val and make segmentation
    merged_nodes, _, _ = vigra.analysis.relabelConsecutive(merged_nodes, start_label=seg_min, keep_zeros=False)
    return nrag.projectScalarNodeDataToPixels(rag, merged_nodes, n_threads)


def postprocess_with_watershed(ds, mc_segmentation, inp_id, size_threshold=500, invert_hmap=False):
    hmap = ds.inp(inp_id)
    assert hmap.shape == mc_segmentation.shape, "%s, %s" % (str(hmap.shape), str(mc_segmentation.shape))

    if invert_hmap:
        hmap = 1. - hmap

    postprocessed = mc_segmentation.copy()
    postprocessed = postprocessed.astype('uint32')
    # need to get rid of 0's, because they correspond to unclaimed territoty
    if postprocessed.min() == 0:
        postprocessed += 1

    # find the ids to merge (smaller than size threshold)
    segment_ids, segment_sizes = np.unique(postprocessed, return_counts=True)
    merge_ids = segment_ids[segment_sizes < size_threshold]

    print "Merge with according to size threshold %i:" % size_threshold
    print "Merging %i / %i segments" % (len(merge_ids), len(segment_ids))

    # mask out the merge-ids
    mask = np.ma.masked_array(postprocessed, mask=np.in1d(postprocessed, merge_ids))
    mask = mask.mask
    postprocessed[mask] = 0

    def pp_z(z):
        ws_z, _ = vigra.analysis.watershedsNew(hmap[z].astype('float32'), seeds=postprocessed[z])
        postprocessed[z] = ws_z

    with futures.ThreadPoolExecutor(max_workers=8) as tp:
        tasks = [tp.submit(pp_z, z) for z in xrange(postprocessed.shape[0])]
        [t.result() for t in tasks]

    postprocessed, _, _ = vigra.analysis.relabelConsecutive(postprocessed, start_label=1, keep_zeros=False)
    return postprocessed


# merge segments that are full enclosed (== have only a single neighboring segment)
def merge_fully_enclosed(mc_segmentation, n_threads=-1):
    rag = nrag.gridRag(mc_segmentation, numberOfThreads=n_threads)
    ufd = nifty.ufd.ufd(rag.numberOfNodes)

    for node_id in xrange(rag.numberOfNodes):
        adjacent_nodes = rag.nodeAdjacency(node_id)
        if len(adjacent_nodes) == 1:
            ufd.merge(node_id, adjacent_nodes[0])

    new_node_labels = ufd.elementLabeling()
    return nrag.projectScalarNodeDataToPixels(rag, new_node_labels, numberOfThreads=n_threads)


if __name__ == '__main__':
    from DataSet import load_dataset
    ds = load_dataset(
        '/home/constantin/Work/home_hdd/cache/cremi_new/sample_A_test'
    )
    seg = vigra.readHDF5(
        '/home/constantin/Work/home_hdd/results/cremi/affinity_experiments/mc_nolearn/gs/sampleA_top_0_.h5',
        'volumes/labels/neuron_ids'
    )
    seg_pp = postprocess_with_watershed(ds, seg, 1)
    from volumina_viewer import volumina_n_layer
    volumina_n_layer(
        [ds.inp(0).astype('float32'), seg, seg_pp]
    )
