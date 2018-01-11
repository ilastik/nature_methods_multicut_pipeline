import np as np
import vigra


def sparse_lifted_edges_and_features(ds, seg_id, prior_threshold=.5):
    rag = ds.rag(seg_id)
    n_nodes = rag.numberOfNodes
    with_priors = np.zeros(n_nodes, dtype='bool')

    pixel_features = [ds.inp(1), ds.inp(2)]
    seg = ds.seg(seg_id)

    for pf in pixel_features:

        # get segments that have a prior in this pixel feature
        acc_feats = vigra.analysis.extractRegionFeatures(pf, seg, ["maximum"])
        acc_max = acc_feats["maximum"]
        indices = acc_max > prior_threshold

        # matched = np.logical_and(with_priors, indices)
        # if np.sum(matched) > 0:
        #     print "have both priors:", np.sum(matched)

        # add nodes to the priors
        with_priors = np.logical_or(with_priors, indices)

    # find all edges which have associated priors
    prior_pairs = np.outer(with_priors, with_priors)
    for i in range(n_nodes):
        prior_pairs[i, i] = 0

    # only take the edges, where both nodes have a high prior
    chosen = np.where(prior_pairs > 0)
    lifted_edges = np.zeros((len(chosen[0]), 2), dtype='uint64')
    lifted_edges[:, 0] = chosen[0]
    lifted_edges[:, 1] = chosen[1]
    lifted_features = np.concatenate([from_pixels_to_edges(lifted_edges, pf, seg, prior_threshold=prior_threshold)
                                      for pf in pixel_features],
                                     axis=1)
    # sanity check
    assert lifted_features.shape[0] == lifted_edges.shape[0]
    return lifted_edges, lifted_features


def from_pixels_to_edges(uv, pf, seg):
    stats = ['maximum', 'minimum', 'mean', 'variance', 'quantiles']
    extractor = vigra.analysis.extractRegionFeatures(pf, seg, stats)
    node_feats = np.concatenate([extractor[stat_name][:, None].astype('float32') if extractor[stat_name].ndim == 1
                                 else extractor[stat_name].astype('float32') for stat_name in stats],
                                axis=1)
    u_feat = node_feats[uv[:, 0]]
    v_feat = node_feats[uv[:, 1]]
    edge_feats = np.concatenate([np.minimum(u_feat, v_feat),
                                 np.maximum(u_feat, v_feat),
                                 np.abs(u_feat, v_feat)],
                                axis=1)
    return edge_feats


def mito_features(ds, seg_id, extra_uv):
    seg = ds.seg(seg_id)
    mito_prob = ds.inp(3)
    feats = from_pixels_to_edges(extra_uv, mito_prob, seg)
    return feats
