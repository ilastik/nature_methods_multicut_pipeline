import numpy as np
import vigra
import os

from .. import RandomForest, local_feature_aggregator
from .. import DataSet, ExperimentSettings
from .. import edges_to_volume
from .. import get_ignore_edge_ids

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty.ground_truth as ngt
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.ground_truth as ngt
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi.ground_truth as ngt
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


# returns a mask that identifies all the edges between superpixels that are
# active in 'segmentation'
def edgemask_from_segmentation(ds, seg_id, segmentation):
    rag = ds.rag(seg_id)
    uv_ids = rag.uvIds()
    node_projection = nrag.gridRagAccumulateLabels(rag, segmentation)
    return node_projection[uv_ids[:, 0]] != node_projection[uv_ids[:, 1]]


# find the overlapping synapse ids for each node
def synapse_node_labels(ds, seg_id, synapse_gt_path, synapse_gt_key):
    # load the synapse gt from file
    synapse_gt = vigra.readHDF5(synapse_gt_path, synapse_gt_key)
    assert synapse_gt.shape == ds.shape

    # get the uv-ids and number of nodes
    uv_ids = ds.uv_ids(seg_id)
    n_nodes = uv_ids.max() + 1

    # compute the overlaps of synapse groundtruth segments with superpixels
    overlaps = ngt.Overlap(n_nodes - 1, ds.seg(seg_id), synapse_gt)

    # extract the max overlaps for each superpixel, ignoring overlaps with 0
    synapse_labels_to_nodes = [overlaps.overlapArraysNormalized(node_id)[0] for node_id in xrange(n_nodes)]
    return synapse_labels_to_nodes


# TODO segmentation
# mask edge labels that should not be used in learning
# NOTE for now we mask the z-edges, but it might be worth a try to use them too
# though in that case two different random forests might be more appropriate
def synapse_edge_mask(ds, seg_id, seg_path, seg_key, with_defects=False):

    segmentation = vigra.readHDF5(seg_path, seg_key)
    uv_ids = ds.uv_ids(seg_id)
    mask = np.ones(len(uv_ids), dtype=bool)

    # we mask the z-edges, because they are not appropriate for determining synapses
    # TODO we could also try this with z-edges included
    mask[ds.edge_indications(seg_id) == 0] = False

    # mask with segmentation
    mask[np.logical_not(edgemask_from_segmentation(ds, seg_id, segmentation))] = False

    # we mask edges that are ignored due to defects
    if with_defects:
        mask[get_ignore_edge_ids(seg_id)] = False

    return mask


# TODO cacher ?!
# project the synapse groundtruth to superpixels and
# then determine for each edge whether it corresponds to a synapse edge
# edges are only marked as synapse edge if the adjacent superpixels are
# mapped to the same synapse id (TODO check different projection strategies)
def synapse_edge_labels(ds, seg_id, synapse_labels_to_nodes):

    # get the uv-ids
    uv_ids = ds.uv_ids(seg_id)

    # compute the labels and mask for edges
    labels = np.zeros(len(uv_ids), dtype='uint8')

    # TODO vectorize
    # loop over the edges and set the label and mask value according to:
    # nodes connected by edge contain the same syn id -> 1
    # nodes contain only ignore label or no overlap at all -> 0
    # nodes
    for edge_id in xrange(len(uv_ids)):
        u, v = uv_ids[edge_id]
        syns_u, syns_v = synapse_labels_to_nodes[u], synapse_labels_to_nodes[v]
        matches = syns_u[np.in1d(syns_u, syns_v)]

        # check if we have matching synapse ids
        if matches.size:
            # check if we have a match different from the ignore id
            # if not, we continue
            if len(matches) == 1 and 0 in matches:
                continue

            # if we have a non-zero match, we set the edge label to 1
            labels[edge_id] = 1

    return labels


# TODO implement
def synapse_edge_labels_from_partner_annotations(ds, seg_id):
    pass


# check the synapse edge labels for debugging
def view_synapse_edge_labels(
    ds,
    seg_id,
    label,
    mask,
    synapse_gt_path,
    synapse_gt_key,
    synapse_labels_to_nodes=None
):
    from volumina_viewer import volumina_n_layer

    rag = ds.rag(seg_id)
    edge_indications = ds.edge_indications(seg_id)
    labels_show = np.zeros_like(label, dtype='uint32')

    # TODO adjust here if there are more label types
    labels_show[label == 0] = 1
    labels_show[label == 1] = 2
    labels_show[np.logical_not(mask)] = 5

    labels_show_xy = labels_show.copy()
    labels_show_xy[edge_indications == 0] = 0

    labels_show_z = labels_show.copy()
    labels_show_z[edge_indications == 1] = 0

    edge_vol_xy = edges_to_volume(rag, labels_show_xy)
    edge_vol_z = edges_to_volume(rag, labels_show_z)

    raw = ds.inp(0).astype('float32')
    seg = ds.seg(seg_id)
    syn_gt = vigra.readHDF5(synapse_gt_path, synapse_gt_key)

    data = [raw, seg, syn_gt, edge_vol_xy, edge_vol_z]
    labels = ['raw', 'seg', 'syn_gt', 'labels_xy', 'labels_z']

    # project the node labels back to the volume, if they are passed
    if synapse_node_labels is not None:
        node_labels = np.array([np.sum(nl) for nl in synapse_labels_to_nodes])
        node_vol = nrag.projectScalarNodeDataToPixels(rag, node_labels, ExperimentSettings().n_threads)
        data.append(node_vol)
        labels.append('syn_node_labels')

    volumina_n_layer(data, labels)


# learn random forest that predicts synapses for a given edge
# this can be learned from multiple datasets with corresponding synapse groundtruth
# this also needs a list of paths and keys for the volumetric synapse groundtruths
# if you want to only learn on a single dataset, don't pass 'trainsets', 'synapse_gt_paths'
# and 'synapse_gt_keys' as a list
# features determines which edge features are used (see 'local_feature_aggregator' in 'EdgeRF.py')
def learn_synapse_rf(
    trainsets,
    seg_id,
    synapse_gt_paths,
    synapse_gt_keys,
    segmentation_paths,
    segmentation_keys,
    feature_list,
    with_defects=False
):

    # we support either training on a single dataset or on a list of datasets
    # if we only have a single one, we need to modify the input here to have lists
    if isinstance(trainsets, DataSet):
        assert isinstance(synapse_gt_paths, str)
        assert isinstance(synapse_gt_keys, str)
        trainsets = [trainsets]
        synapse_gt_paths = [synapse_gt_paths]
        synapse_gt_keys = [synapse_gt_keys]

    assert len(trainsets) == len(synapse_gt_paths)
    assert len(trainsets) == len(synapse_gt_keys)

    # check if this random forest is already cached
    # we only cache here, if the 'rf_cache_folder' is not set to None
    cache_folder = ExperimentSettings().rf_cache_folder
    if cache_folder is not None:
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        trainstr = "_".join([ds.ds_name for ds in trainsets]) + "_" + str(seg_id)
        paramstr = "_".join(feature_list)
        rf_folder = os.path.join(cache_folder, "synapserf_%s" % trainstr)
        rf_name = "synapserf_%s_%s" % (trainstr, paramstr)
        if len(rf_name) > 255:
            rf_name = str(hash(rf_name))

        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)

        rf_path   = os.path.join(rf_folder, rf_name)
        if RandomForest.is_cached(rf_path):
            print "Loading synapse random forest from:"
            print rf_path
            return RandomForest.load_from_file(rf_path, 'rf_synapse', ExperimentSettings().n_threads)

    features = []
    labels = []

    # iterate over the trainsets and extract the features and labels for the edges for each dataset
    for ii, ds in enumerate(trainsets):
        synapse_labels_to_nodes = synapse_node_labels(ds, seg_id, synapse_gt_paths[ii], synapse_gt_keys[ii])
        # TODO change function for labels here
        label = synapse_edge_labels(ds, seg_id, synapse_labels_to_nodes)
        mask = synapse_edge_mask(ds, seg_id, segmentation_paths[ii], segmentation_keys[ii], with_defects)

        # check the edge labels for debugging
        if False:
            view_synapse_edge_labels(
                ds, seg_id, label, mask, synapse_gt_paths[ii], synapse_gt_keys[ii], synapse_labels_to_nodes
            )
            quit()

        features.append(
            local_feature_aggregator(ds, seg_id, feature_list, ExperimentSettings().anisotropy_factor, True)[mask]
        )
        labels.append(label[mask])

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # learn the random forest
    rf = RandomForest(features, labels, ExperimentSettings().n_trees, ExperimentSettings().n_threads)

    # cache the rf if caching is activated
    if ExperimentSettings().rf_cache_folder is not None:
        rf.write(rf_path, 'rf_synapse')

    return rf


# TODO cache ?!
# predict the synapse edge probabilities for the test-set,
# with random forest learned on the train-sets
def predict_synapse_edge_probabilities(
    trainsets,
    synapse_gt_paths,
    synapse_gt_keys,
    segmentation_train_paths,
    segmentation_train_keys,
    ds_test,
    segmentation_test_path,
    segmentation_test_key,
    seg_id_train,
    seg_id_test,
    feature_list,
    with_defects=False
):

    # get the random forest, features
    # and predict the synapse edge probabilities
    rf = learn_synapse_rf(
        trainsets,
        seg_id_train,
        synapse_gt_paths,
        synapse_gt_keys,
        segmentation_train_paths,
        segmentation_train_keys,
        feature_list,
        with_defects
    )

    features = local_feature_aggregator(
        ds_test,
        seg_id_test,
        feature_list,
        ExperimentSettings().anisotropy_factor,
        True
    )

    syn_probs = rf.predict_probabilities(features)[:, 1]

    # get the edge mask
    mask = synapse_edge_mask(
        ds_test,
        seg_id_test,
        segmentation_test_path,
        segmentation_test_key,
        with_defects
    )
    return syn_probs, mask
