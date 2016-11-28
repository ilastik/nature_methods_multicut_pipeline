import vigra
import numpy as np

from volumina_viewer import volumina_n_layer

def thin_by_ws(gt_labels, probs):

    gt_thin = np.zeros_like(gt_labels, dtype = np.uint32)
    offset = 0
    for z in xrange(gt_labels.shape[2]):
        hmap = vigra.filters.gaussianSmoothing( probs[:,:,z], 2.)
        seeds = vigra.analysis.labelImageWithBackground( gt_labels[:,:,z] )
        gt_thin[:,:,z], _ = vigra.analysis.watershedsNew(hmap,
                seeds = seeds)
        gt_thin[:,:,z][ gt_thin[:,:,z] != 0 ] += offset
        offset = gt_thin[:,:,z].max()

    return gt_thin - 1

# clean segments, that are completely embededd
def clean_isolated(gt):
    import vigra.graphs as vgraph

    rag_global = vgraph.regionAdjacencyGraph( vgraph.gridGraph(gt.shape[0:3]), gt)

    node_to_node = np.concatenate(
            [ np.arange(rag_global.nodeNum, dtype = np.uint32)[:,None] for _ in range(2)]
            , axis = 1 )

    for z in xrange(gt.shape[2]):
        rag_local = vgraph.regionAdjacencyGraph( vgraph.gridGraph(gt.shape[0:2]), gt[:,:,z])
        for node in rag_local.nodeIter():
            neighbour_nodes = []
            for nnode in rag_local.neighbourNodeIter(node):
                neighbour_nodes.append(nnode)
            if len(neighbour_nodes) == 1:
                node_coordinates = np.where(gt == node.id)
                if not 0 in node_coordinates[0] and not 511 in node_coordinates[0] and not 0 in node_coordinates[1] and not 511 in node_coordinates[1]:
                    node_to_node[node.id] = neighbour_nodes[0].id

    gt_cleaned = rag_global.projectLabelsToBaseGraph(node_to_node)[:,:,:,0]

    return gt_cleaned


if __name__ == '__main__':

    gt = vigra.readHDF5(
            "/home/consti/Work/nature_experiments/isbi12_data/groundtruth/gt_layerwise.h5",
            "gt")

    gt_labels = np.squeeze( vigra.impex.readVolume(
            "/home/consti/Work/nature_experiments/isbi12_data/groundtruth/train-labels.tif") )
    gt_labes = np.array( gt_labels)

    raw = vigra.readHDF5(
            "/home/consti/Work/nature_experiments/isbi12_data/raw/train-volume.h5",
            "data")

    probs = vigra.readHDF5(
            "/home/consti/Work/nature_experiments/isbi12_data/probabilities/interceptor_train.h5",
            "data")

    gt_thin = thin_by_ws(gt_labels, probs)
    gt_clean = clean_isolated(gt_thin)

    vigra.writeHDF5( gt_clean,
            "/home/consti/Work/nature_experiments/isbi12_data/groundtruth/gt_cleaned.h5",
            "data")

    #volumina_n_layer( [raw, gt, gt_thin, gt_clean] )
