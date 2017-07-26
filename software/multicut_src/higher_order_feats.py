import vigra
import numpy as np
from functools import partial
from itertools import permutations
import os

from ExperimentSettings import ExperimentSettings
from EdgeRF import RandomForest, local_feature_aggregator, local_feature_aggregator_with_defects
from tools import replace_from_dict, cacher_hdf5

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty.cgp as ncgp
    import nifty.graph.rag as nrag
    # import nifty.segmentation as nseg
except ImportError:
    try:
        import nifty_with_cplex.cgp as ncgp
        import nifty_with_cplex.graph.rag as nrag
        # import nifty_with_cplex.segmentation as nseg
    except ImportError:
        try:
            import nifty_with_gurobi.cgp as ncgp
            import nifty_with_gurobi.graph.rag as nrag
            # import nifty_with_gurobi.segmentation as nseg
        except ImportError:
            raise ImportError("No valid nifty version was found.")


def _get_cgp(seg, rag):
    # get the segmentation in this slice and map it
    # to a consecutive labeling starting from 1
    seg_z, _, mapping = vigra.analysis.relabelConsecutive(
        seg,
        keep_zeros=False,
        start_label=1
    )
    reverse_mapping = {new: old for old, new in mapping.iteritems()}
    assert seg_z.min() == 1

    # construct the topological grid
    tgrid = ncgp.TopologicalGrid2D(seg_z)
    return tgrid, reverse_mapping


def get_junctions(seg, graph):

    tgrid, reverse_mapping = _get_cgp(seg, graph)

    # get the bounding relations
    cell_bounds = tgrid.extractCellsBounds()

    # junctions (0-cells) to faces (1-cells)
    junctions_to_faces = np.array(cell_bounds[0])

    # check if we have actual 4 junctions, which are not supported yet
    have_4_junction = junctions_to_faces[:, -1] != 0
    # if we have any 4 junctions, drop them
    if have_4_junction.any():
        # raise RuntimeError("We have %i four junctions, giving up!" % np.sum(have_4_junction != 0))
        print "Dropping %i 4-junctions" % np.sum(have_4_junction)
        junctions_to_faces = junctions_to_faces[np.logical_not(have_4_junction)]

    # drop last axis
    junctions_to_faces = junctions_to_faces[:, :-1]
    assert junctions_to_faces.shape[1] == 3

    # there should not be two junctions
    assert (junctions_to_faces != 0).all()

    # get the mapping from faces to edges

    # first, get the uv ids corresponding to the lines / faces
    # (== 1 cells), and map them back to the original ids
    line_uv_ids = np.array(cell_bounds[1])
    line_uv_ids = np.sort(
        replace_from_dict(line_uv_ids, reverse_mapping),
        axis=1
    )

    # find the corresponding edge ids
    edge_ids = graph.findEdges(line_uv_ids)
    assert (edge_ids != -1).all()
    # need to add offset due to cgp 1 based counting
    edge_ids = np.insert(edge_ids, 0, 0)

    # map face ids to edge ids
    junctions_to_edges = edge_ids[junctions_to_faces]

    # return the number of junctions and the mapping of junctions to faces
    return junctions_to_edges


@cacher_hdf5()
def get_xy_junctions(ds, seg_id, with_defects=False):
    seg = ds.seg(seg_id)
    if with_defects:
        assert False, "Not implemented"  # TODO
    else:
        graph = ds.rag(seg_id)

    junctions_to_edges = []
    for z in xrange(seg.shape[0]):
        z_junctions = get_junctions(seg[z], graph)
        junctions_to_edges.append(z_junctions)

    return np.concatenate(junctions_to_edges)


# TODO include dedicated junction feats
# accumulate the higher order features
def higher_order_feature_aggregator(
    ds,
    seg_id,
    higher_order_feat_list,
    edge_feat_list,
    anisotropy_factor,
    use_2d,
    with_defects=False
):
    for feat in higher_order_feat_list:
        assert feat in ('edge_feats',), feat

    junction_feats = []
    if 'edge_feats' in higher_order_feat_list:
        junction_feats.append(
            junction_feats_from_edge_feats(ds, seg_id, edge_feat_list, anisotropy_factor, use_2d)
        )

    return np.concatenate(junction_feats, axis=1)


# make permuted features from list of junction features
# FIXME this is copied from thorsten without exactly understanding what is going on here...
def make_permuted_features(feat_list):

    feats = []

    f_stacked = np.stack(feat_list)

    feats.append(np.min(f_stacked, axis=0))
    feats.append(np.max(f_stacked, axis=0))
    sorted_args = np.argsort(f_stacked, axis=0)
    feats.append(sorted_args[0, ...])
    feats.append(sorted_args[1, ...])
    feats.append(sorted_args[2, ...])
    feats.append(np.concatenate(feat_list, axis=1))

    return np.concatenate(feats, axis=1)


# map edge features to junctions
def junction_feats_from_edge_feats(
    ds,
    seg_id,
    feature_list,
    anisotropy_factor,
    use_2d,
    with_defects=False
):

    edge_feats = local_feature_aggregator_with_defects(ds, seg_id, feature_list, anisotropy_factor, use_2d) \
        if with_defects else local_feature_aggregator(ds, seg_id, feature_list, anisotropy_factor, use_2d)

    xy_junctions = get_xy_junctions(ds, seg_id, with_defects)

    # get the features for all junction permutations
    if ExperimentSettings().use_junction_permutations:
        permutation_feats = []
        for perm in permutations([0, 1, 2]):
            feat_list = [
                edge_feats[xy_junctions[:, i], :] for i in perm
            ]
            permutation_feats.append(make_permuted_features(feat_list))
        return np.concatenate(permutation_feats, axis=0)

    # get simple junction features without permuting
    else:
        junction_feats = np.concatenate(
            [edge_feats[xy_junctions[:, i]] for i in range(xy_junctions.shape[1])],
            axis=1
        )
        assert len(xy_junctions) == len(junction_feats)
        assert junction_feats.shape[1] == 3 * edge_feats.shape[1]

        return junction_feats


# TODO
# dedicated features for junctions
# TODO


def edge_state_to_junction_label(edge_states):
    # find the correct groundtruth
    junction_gt = np.zeros(len(edge_states), dtype='uint8')
    has_state = np.zeros(len(edge_states), dtype='uint8')

    # combination (0,0,0) -> 0, (0,1,1) -> 1, (1,0,1) -> 2, (1,1,0) -> 3, (1,1,1) -> 4
    valid_states = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    for i, state in enumerate(valid_states):
        where_state = np.where((edge_states == state).all(axis=1))
        junction_gt[where_state] = i
        has_state[where_state] += 1
    assert (has_state == 1).all()
    return junction_gt


# junction labels from groundtruth
def junction_groundtruth(ds, seg_id, with_defects=False):
    rag = ds.rag(seg_id)
    gt = ds.gt()

    uv_ids = rag.uvIds()
    node_gt = nrag.gridRagAccumulateLabels(rag, gt)

    xy_junctions = get_xy_junctions(ds, seg_id, with_defects)

    # find the states for all three edges involved in a junction
    edge_states = np.array([
        node_gt[uv_ids[xy_junctions[:, i]][:, 0]] !=
        node_gt[uv_ids[xy_junctions[:, i]][:, 1]] for i in range(xy_junctions.shape[1])
    ]).transpose()
    assert edge_states.shape == xy_junctions.shape, "%s, %s" % (str(edge_states.shape), str(xy_junctions.shape))

    # make sure we have no illegal combinations
    assert (np.sum(edge_states, axis=1) != 1).all()

    # TODO z junction gt ?

    # get the groundtruth for all junction permutations
    if ExperimentSettings().use_junction_permutations:
        permutation_gt = np.concatenate(
            [edge_states[:, permute] for permute in permutations([0, 1, 2])],
            axis=0
        )
        return edge_state_to_junction_label(permutation_gt)

    # get simple junction groundtruth without permutations
    else:
        return edge_state_to_junction_label(edge_states)


# TODO split in function for xy - and z once necessary
# learn random forests from higher order feats
def learn_higher_order_rf(
    trainsets,
    seg_id,
    feature_aggregator,
    trainstr,
    paramstr,
    with_defects=False
):
    cache_folder = ExperimentSettings().rf_cache_folder
    # check if already cached
    if cache_folder is not None:  # we use caching for the rf => look if already exists
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        rf_folder = os.path.join(cache_folder, "junction_rf" + trainstr)
        rf_name = "rf_" + "_".join([trainstr, paramstr]) + ".h5"
        if len(rf_name) > 255:
            rf_name = str(hash(rf_name))
        if not os.path.exists(rf_folder):
            os.mkdir(rf_folder)

        rf_path   = os.path.join(rf_folder, rf_name)
        if RandomForest.is_cached(rf_path):
            print "Loading random forest from:"
            print rf_path
            return RandomForest.load_from_file(rf_path, 'rf_junctions', ExperimentSettings().n_threads)

    # TODO junction masking
    features_train = []
    labels_train = []

    # iterate over the datasets in our traninsets
    for ds in trainsets:

        assert ds.has_gt
        features = feature_aggregator(ds, seg_id)
        labels = junction_groundtruth(ds, seg_id, with_defects)
        assert len(features) == len(labels)

        # TODO mask once implemented
        features_train.append(features)
        labels_train.append(labels)

    features_train = np.concatenate(features_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)

    rf = RandomForest(
        features_train,
        labels_train,
        n_trees=ExperimentSettings().n_trees,
        n_threads=ExperimentSettings().n_threads
    )

    if cache_folder is not None:
        rf.write(rf_path, 'rf_junctions')

    return rf


def learn_and_predict_higher_order_rf(
    trainsets,
    ds_test,
    seg_id_train,
    seg_id_test,
    edge_feat_list,
    higher_order_feat_list=('edge_feats',),
    with_defects=False
):

    assert not with_defects  # not implemented yet

    # for only a single ds, put it in a list
    if not isinstance(trainsets, list):
        trainsets = [trainsets]

    # strings for caching
    # str for all relevant params
    paramstr = "_".join(
        ["_".join(edge_feat_list), "_".join(higher_order_feat_list),
         str(ExperimentSettings().n_trees), str(ExperimentSettings().use_junction_permutations)]
    )
    teststr  = ds_test.ds_name + "_" + str(seg_id_test)
    trainstr = "_".join([ds.ds_name for ds in trainsets]) + "_" + str(seg_id_train)

    feature_aggregator = partial(
        higher_order_feature_aggregator,
        higher_order_feat_list=higher_order_feat_list,
        edge_feat_list=edge_feat_list,
        anisotropy_factor=ExperimentSettings().anisotropy_factor,
        use_2d=ExperimentSettings().use_2d,
        with_defects=with_defects
    )

    # we cache this in the ds_test cache folder
    # if caching random forests is activated (== rf_cache_folder is not None)
    if ExperimentSettings().rf_cache_folder is not None:  # cache-folder exists => look if we already have a prediction

        pred_name = "junction_prediction_" + "_".join([trainstr, teststr, paramstr]) + ".h5"
        if len(pred_name) >= 256:
            pred_name = str(hash(pred_name[:-3])) + ".h5"
        pred_path = os.path.join(ds_test.cache_folder, pred_name)

        # see if the rf is already learned and predicted, otherwise learn it
        if os.path.exists(pred_path):
            print "Loading prediction from:"
            print pred_path
            return vigra.readHDF5(pred_path, 'data')

    # learn the rf
    rf = learn_higher_order_rf(trainsets, seg_id_train, feature_aggregator, trainstr, paramstr, with_defects)

    # get the test features
    features_test  = feature_aggregator(ds_test, seg_id_test)

    # predict
    junction_probs = rf.predict_probabilities(features_test)

    # if we use the junction permutations, we need to map back to the original permutation (0, 1, 2)
    if ExperimentSettings().use_junction_permutations:
        junction_probs = permutations_to_junction_probs(junction_probs)

    # TODO cache

    return junction_probs


def permutations_to_junction_probs(junction_probs):
    # we need to extract the corresponding probabilities for all permutations
    perms = [pp for pp in permutations([0, 1, 2])]
    n_perms = len(perms)

    # make sure the number of permutation divides the length of probability vector
    assert len(junction_probs) % n_perms == 0
    n_junctions = len(junction_probs) / n_perms

    states = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    assert len(states) == junction_probs.shape[1]

    actual_probs = np.zeros((n_junctions, len(states)), dtype='float32')
    for i, pp in enumerate(perms):
        perm_start, perm_stop = i * n_junctions, (i + 1) * n_junctions
        if i + 1 == n_perms:
            assert perm_stop == len(junction_probs)
        this_probs = junction_probs[perm_start:perm_stop, :]

        # map the states in the current permutation to the actual states
        permuted_states = np.array([
            tuple(np.array(s)[list(pp)]) for s in states
        ])
        state_permutation = [
            np.where((s == permuted_states).all(axis=1))[0][0] for s in states
        ]

        actual_probs[:] += this_probs[:, state_permutation]

    actual_probs /= n_perms
    return actual_probs


# TODO fancy weighting
def junction_predictions_to_costs(junction_probs, beta=.5):

    # scale the probabilities
    # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
    p_min = 0.001
    p_max = 1. - p_min

    junction_probs = (p_max - p_min) * junction_probs + p_min

    junction_costs = np.log((1. - junction_probs) / junction_probs) + np.log((1. - beta) / beta)
    return junction_costs


def project_junction_probs_to_edges(ds, seg_id, junction_probs):
    states = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    assert len(states) == junction_probs.shape[1]

    n_edges = len(ds.uv_ids(seg_id))

    junctions_to_edges = get_xy_junctions(ds, seg_id)
    assert len(junctions_to_edges) == len(junction_probs)
    edges_with_junctions, edge_count = np.unique(junctions_to_edges, return_counts=True)
    assert edges_with_junctions[-1] < n_edges

    edge_probs = np.zeros(n_edges, dtype='float32')

    for j_id, probs in enumerate(junction_probs):
        edges = junctions_to_edges[j_id]
        # write probabilities to the edges according to the states of this junction
        for s_id in range(1, len(states)):
            active_edges = np.array(states[s_id]) == 1
            edge_probs[edges[active_edges]] += probs[s_id]

    # normalize the projected edge probabilities
    edge_count *= 3  # each edge has 3 probabilities for state 1 per junction
    edge_probs[edges_with_junctions] /= edge_count

    # make sure that edges without junctions (z-edges) have no prob.
    no_junction_edges = np.ones(n_edges, dtype=bool)
    no_junction_edges[edges_with_junctions] = False
    assert (edge_probs[no_junction_edges] == 0).all()

    return edge_probs
