import vigra
import h5py
import numpy as np
from concurrent import futures
from tools import cacher_hdf5, cache_name, get_unique_rows, find_matching_row_indices, find_matching_indices

from MCSolverImpl import weight_z_edges, weight_all_edges, weight_xyz_edges
from ExperimentSettings import ExperimentSettings
from DataSet import _topology_features_impl

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty
except ImportError:
    try:
        import nifty_with_cplex as nifty  # conda version build with cplex
    except ImportError:
        try:
            import nifty_with_gurobi as nifty  # conda version build with gurobi
        except ImportError:
            raise ImportError("No valid nifty version was found.")
import nifty.graph.rag as nrag
import nifty.ground_truth as ngt

#
# Modified Adjacency
#


@cacher_hdf5()
def defects_to_nodes(ds, seg_id):

    if ds.external_defect_mask_path is None:
        return []

    defect_mask = ds.defect_mask()
    seg = ds.seg(seg_id)
    assert seg.shape == defect_mask.shape

    # TODO test the heuristics
    def defects_to_nodes_z(z):
        defect_mask_z = defect_mask[:, :, z]
        where_defect = defect_mask_z > 0
        n_defect_pix = np.sum(where_defect)
        completely_defected = False
        if n_defect_pix > 0:
            seg_z = seg[:, :, z]
            # heuristics to determine completely defected slice
            # -> all nodes are set to defected and this slice will be replaced in postprocessing
            if n_defect_pix > .5 * defect_mask_z.size:
                defect_nodes_z = np.unique(seg_z)
                completely_defected = True
            else:
                defect_nodes_z = np.unique(seg_z[where_defect])
            if ds.has_seg_mask and ExperimentSettings().ignore_seg_value in defect_nodes_z:
                defect_nodes_z = defect_nodes_z[defect_nodes_z != ExperimentSettings().ignore_seg_value]
        else:
            defect_nodes_z = []

        return list(defect_nodes_z), len(defect_nodes_z) * [z], completely_defected

    completely_defected_slices = []
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = []
        for z in xrange(seg.shape[2]):
            tasks.append(executor.submit(defects_to_nodes_z, z))
        defect_nodes = []
        nodes_z      = []
        for fut in tasks:
            nodes, zz, completely_defected = fut.result()
            if nodes:
                defect_nodes.extend(nodes)
                nodes_z.extend(zz)
                if completely_defected:
                    completely_defected_slices.append(zz[0])

    assert len(defect_nodes) == len(nodes_z)
    # only mask the dataset as having defects, if we actually found any defected nodes
    if defect_nodes:
        ds._has_defects = True
        ds.save()

    defect_nodes = np.array(defect_nodes, dtype='uint32')
    nodes_z = np.array(nodes_z, dtype='uint32')

    save_path = cache_name("defects_to_nodes", "dset_folder", False, False, ds, seg_id)
    vigra.writeHDF5(nodes_z, save_path, 'nodes_z')
    vigra.writeHDF5(completely_defected_slices, save_path, "completely_defected_slices")

    return defect_nodes


# this is very hacky due to stupid caching...
# we calculate everything with modified adjacency and then load the things with individual functions

def get_defect_node_z(ds, seg_id):
    defects_to_nodes(ds, seg_id)
    save_path = cache_name("defects_to_nodes", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(save_path, 'nodes_z')


def get_completely_defected_slices(ds, seg_id):
    defects_to_nodes(ds, seg_id)
    save_path = cache_name("defects_to_nodes", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(save_path, "completely_defected_slices")


def get_delete_edges(ds, seg_id):
    modified_adjacency(ds, seg_id)
    if not ds.has_defects:
        return np.array([])
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(mod_save_path, "delete_edges")


def get_delete_edge_ids(ds, seg_id):
    modified_adjacency(ds, seg_id)
    if not ds.has_defects:
        return np.array([])
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(mod_save_path, "delete_edge_ids")


def get_ignore_edges(ds, seg_id):
    modified_adjacency(ds, seg_id)
    if not ds.has_defects:
        return np.array([])
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(mod_save_path, "ignore_edges")


def get_ignore_edge_ids(ds, seg_id):
    modified_adjacency(ds, seg_id)
    if not ds.has_defects:
        return np.array([])
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(mod_save_path, "ignore_edge_ids")


def get_skip_edges(ds, seg_id):
    modified_adjacency(ds, seg_id)
    if not ds.has_defects:
        return np.array([])
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(mod_save_path, "skip_edges")


def get_skip_ranges(ds, seg_id):
    modified_adjacency(ds, seg_id)
    if not ds.has_defects:
        return np.array([])
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(mod_save_path, "skip_ranges")


def get_skip_starts(ds, seg_id):
    modified_adjacency(ds, seg_id)
    if not ds.has_defects:
        return np.array([])
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)
    return vigra.readHDF5(mod_save_path, "skip_starts")


# this should be fast enough for now
def compute_skip_edges_z(
        z,
        seg,
        defect_node_dict):

    def skip_edges_for_nodes(z_up, z_dn, nodes_dn, mask):
        skip_range = z_up - z_dn
        skip_edges, skip_ranges = [], []
        defect_nodes_up = np.array(defect_node_dict.get(z_up, []))

        for node_dn in nodes_dn:
            # find the connected nodes in the upper slice
            mask_dn = seg[:, :, z_dn][mask] == node_dn
            seg_up = seg[:, :, z_up][mask]
            connected_nodes = np.unique(seg_up[mask_dn])
            # if any of the connected nodes are defected go to the next slice
            if defect_nodes_up.size:
                if np.intersect1d(connected_nodes, defect_nodes_up).size:  # check if any of upper nodes is defected
                    skip_edges, skip_ranges = skip_edges_for_nodes(z_up + 1, z_dn, nodes_dn, mask)
                    break
            skip_edges.extend([(node_dn, conn_node) for conn_node in connected_nodes])
            skip_ranges.extend(len(connected_nodes) * [skip_range])

        return skip_edges, skip_ranges

    skip_edges_z = []
    skip_ranges_z = []
    defect_nodes_z = defect_node_dict[z]

    for defect_node in defect_nodes_z:
        # get the mask
        mask = seg[:, :, z] == defect_node
        # find the lower nodes overlapping with the defect in the lower slice
        nodes_dn = np.unique(seg[:, :, z - 1][mask])
        # discard defected nodes in lower slice (if present) because they were already taken care of
        nodes_dn = np.array([n_dn for n_dn in nodes_dn if n_dn not in defect_nodes_z])
        # if we have lower nodes left, we look for skip edges
        if nodes_dn.size:
            skip_edges_u, skip_ranges_u = skip_edges_for_nodes(z + 1, z - 1, nodes_dn, mask)
            skip_edges_z.extend(skip_edges_u)
            skip_ranges_z.extend(skip_ranges_u)

    return skip_edges_z, skip_ranges_z


@cacher_hdf5()
def modified_adjacency(ds, seg_id):

    defect_nodes = defects_to_nodes(ds, seg_id)
    if not ds.has_defects:
        return np.array([0])
    nodes_z      = get_defect_node_z(ds, seg_id)

    # make sure that z is monotonically increasing (not strictly!)
    assert np.all(np.diff(nodes_z.astype(int)) >= 0), "Defected slice index is not increasing monotonically!"
    defect_slices = np.unique(nodes_z)
    defect_node_dict = {int(z): defect_nodes[nodes_z == z].astype('uint32').tolist() for z in defect_slices}

    # FIXME TODO can't do this here once we have individual defect patches
    consecutive_defect_slices = np.split(
        defect_slices, np.where(np.diff(defect_slices) != 1)[0] + 1
    )
    has_lower_defect_list = []
    for consec in consecutive_defect_slices:
        if len(consec) > 1:
            has_lower_defect_list.extend(consec[1:])

    # iterate over the nodes in slices with defects to get delete, ignore and skip edges
    seg = ds.seg(seg_id)
    edge_indications = ds.edge_indications(seg_id)

    delete_edges = []  # the z-edges between defected and non-defected nodes that are deleted from the graph
    delete_edge_ids = []
    ignore_edges = []  # the xy-edges between defected and non-defected nodes, that will be set to maximally repulsive weights

    skip_edges   = []  # the skip edges that run over the defects in z
    skip_ranges  = []  # z-distance of the skip edges
    skip_starts  = []  # starting slices of the skip edges

    # get the delete and ignore edges by checking which uv-ids have at least one defect node
    uv_ids = ds.uv_ids(seg_id)
    defect_uv_indices = find_matching_indices(uv_ids, defect_nodes)
    for defect_index in defect_uv_indices:
        if edge_indications[defect_index]:  # we have a xy edge -> ignore edge
            ignore_edges.append(uv_ids[defect_index])
        else:  # z edge -> delete edge
            delete_edges.append(uv_ids[defect_index])
            delete_edge_ids.append(defect_index)

    delete_edges    = np.array(delete_edges, dtype='uint32')
    delete_edge_ids = np.array(delete_edge_ids, dtype='uint32')

    ignore_edges = np.array(ignore_edges, dtype='uint32')
    # find the ignore edge ids -> corresponding to the ids after delete edges are removed !
    # fist, get the uv ids after removal of uv - edges
    uv_ids = np.delete(uv_ids, delete_edge_ids, axis=0)

    if ignore_edges.size:
        assert ignore_edges.shape[1] == uv_ids.shape[1], "%i, %i" % (ignore_edges.shape[1], uv_ids.shape[1])
        matching = find_matching_row_indices(uv_ids, ignore_edges)
        # make sure that all ignore edges were found
        assert matching.shape[0] == ignore_edges.shape[0]
        # get the correctly sorted the ids
        ignore_edge_ids = matching[:, 0]
        ignore_edge_ids = ignore_edge_ids[matching[:, 1]]
    else:
        ignore_edge_ids = []

    for i, z in enumerate(defect_slices):
        print "Processing slice %i: %i / %i" % (z, i, len(defect_slices))
        # get the skip edges between adjacent slices
        # skip for first or last slice or slice with lower defect
        has_lower_defect = True if z in has_lower_defect_list else False
        if z == 0 or z == seg.shape[2] - 1 or has_lower_defect:
            continue

        skip_edges_z, skip_ranges_z = compute_skip_edges_z(z, seg, defect_node_dict)

        assert len(skip_edges_z) == len(skip_ranges_z)
        skip_edges.extend(skip_edges_z)
        skip_ranges.extend(skip_ranges_z)
        skip_starts.extend(len(skip_edges_z) * [z - 1])

    assert skip_edges, "If we are here, we should have skip edges !"
    skip_edges = np.array(skip_edges, dtype=np.uint32)
    # make the skip edge rows unique
    skip_edges, idx = get_unique_rows(skip_edges, return_index=True)

    skip_ranges = np.array(skip_ranges, dtype="uint32")[idx]
    skip_starts = np.array(skip_starts, dtype="uint32")[idx]

    # if we have a seg mask, the skip edges can have entries connecting the ignore segment with itself, we need to remove these
    if ds.has_seg_mask:
        duplicate_mask = skip_edges[:, 0] != skip_edges[:, 1]
        if not duplicate_mask.all():  # -> we have entries that will be masked out
            # make sure that all duplicates have ignore segment value
            assert (skip_edges[np.logical_not(duplicate_mask)] == ExperimentSettings().ignore_seg_value).all()
            print "Removing duplicate skip edges due to ignore segment label"
            skip_edges = skip_edges[duplicate_mask]
            skip_ranges = skip_ranges[duplicate_mask]
            skip_starts = skip_starts[duplicate_mask]

    assert skip_edges.shape[0] == skip_ranges.shape[0]
    assert skip_starts.shape[0] == skip_ranges.shape[0]

    # reorder the skip edges s.t. skip_starts are monotonically increasing
    sort_indices = np.argsort(skip_starts)
    skip_edges = skip_edges[sort_indices]
    skip_ranges = skip_ranges[sort_indices]
    skip_starts = skip_starts[sort_indices]
    # make sure that z is monotonically increasing (not strictly!)
    assert np.all(np.diff(skip_starts.astype(int)) >= 0), "start index of skip edges must increase monotonically."

    # sort the uv ids in skip edges
    skip_edges = np.sort(skip_edges, axis=1)

    # get the modified adjacency
    # first check if we have any duplicates in the skip edges and uv - ids
    # this can happen if we have a segmentation mask
    matches = find_matching_row_indices(uv_ids, skip_edges)
    if matches.size:
        assert ds.has_seg_mask, "There should only be duplicates in skip edges and uvs if we have a seg mask"
        # make sure that all removed edges are ignore edges
        assert all((skip_edges[matches[:, 1]] == ExperimentSettings().ignore_seg_value).any(axis=1)), "All duplicate skip edges should connect to a ignore segment"

        print "Removing %i skip edges that were duplicates of uv ids." % len(matches)
        # get a mask for the duplicates
        duplicate_mask = np.ones(len(skip_edges), dtype=np.bool)
        duplicate_mask[matches[:, 1]] = False

        # remove duplicates from skip edges, ranges and starts
        skip_edges  = skip_edges[duplicate_mask]
        skip_ranges = skip_ranges[duplicate_mask]
        skip_starts = skip_starts[duplicate_mask]

    # new modified adjacency
    modified_adjacency = np.concatenate([uv_ids, skip_edges])

    # save delete, ignore and skip edges, a little hacky due to stupid caching...
    save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id)

    vigra.writeHDF5(delete_edges, save_path, "delete_edges")
    vigra.writeHDF5(delete_edge_ids, save_path, "delete_edge_ids")

    vigra.writeHDF5(ignore_edges, save_path, "ignore_edges")
    vigra.writeHDF5(ignore_edge_ids, save_path, "ignore_edge_ids")

    vigra.writeHDF5(skip_edges, save_path, "skip_edges")
    vigra.writeHDF5(skip_ranges, save_path, "skip_ranges")
    vigra.writeHDF5(skip_starts, save_path, "skip_starts")

    return modified_adjacency


@cacher_hdf5()
def modified_edge_indications(ds, seg_id):
    modified_indications = ds.edge_indications(seg_id)
    n_edges = modified_indications.shape[0]
    skip_edges   = get_skip_edges(ds, seg_id)
    delete_edge_ids = get_delete_edge_ids(ds, seg_id)
    if not ds.has_defects:
        return modified_indications
    modified_indications = np.delete(modified_indications, delete_edge_ids)
    modified_indications = np.concatenate(
        [modified_indications, np.zeros(skip_edges.shape[0], dtype=modified_indications.dtype)]
    )
    assert modified_indications.shape[0] == n_edges - delete_edge_ids.shape[0] + skip_edges.shape[0]
    return modified_indications


@cacher_hdf5()
def modified_edge_gt(ds, seg_id):
    modified_edge_gt = ds.edge_gt(seg_id)
    skip_edges   = get_skip_edges(ds, seg_id)
    delete_edge_ids = get_delete_edge_ids(ds, seg_id)
    if not ds.has_defects:
        return modified_edge_gt
    modified_edge_gt = np.delete(modified_edge_gt, delete_edge_ids)
    rag = ds.rag(seg_id)
    node_gt = nrag.gridRagAccumulateLabels(rag, ds.gt())  # ExperimentSettings().n_threads)
    skip_gt = (node_gt[skip_edges[:, 0]] != node_gt[skip_edges[:, 1]]).astype('uint8')
    return np.concatenate([modified_edge_gt, skip_gt])


@cacher_hdf5()
def modified_edge_gt_fuzzy(ds, seg_id, positive_threshold, negative_threshold):
    uv_ids = modified_adjacency(ds, seg_id)
    overlaps = ngt.Overlap(
        uv_ids.max(),
        ds.seg(seg_id),
        ds.gt()
    )
    edge_overlaps = overlaps.differentOverlaps(uv_ids)
    edge_gt_fuzzy = 0.5 * np.ones(edge_overlaps.shape, dtype='float32')
    edge_gt_fuzzy[edge_overlaps > positive_threshold] = 1.
    edge_gt_fuzzy[edge_overlaps < negative_threshold] = 0.
    return edge_gt_fuzzy

#
# Modified Features
#


@cacher_hdf5(folder="feature_folder", cache_edgefeats=True)
def modified_edge_features_from_affinity_maps(ds, seg_id, inp_ids, anisotropy_factor, z_direction):
    modified_features = ds.edge_features_from_affinity_maps(seg_id, inp_ids, anisotropy_factor, z_direction)

    skip_edges   = get_skip_edges( ds, seg_id)
    skip_starts  = get_skip_starts(ds, seg_id)
    skip_ranges  = get_skip_ranges(ds, seg_id)
    delete_edge_ids = get_delete_edge_ids(ds, seg_id)

    if not ds.has_defects:
        return modified_features

    # delete features for delete edges
    modified_features = np.delete(modified_features, delete_edge_ids, axis=0)

    # get features for skip edges
    seg = ds.seg(seg_id)
    lower_slices  = np.unique(skip_starts)
    skip_edge_pairs_to_slice   = {z: skip_edges[skip_starts == z] for z in lower_slices}
    skip_edge_indices_to_slice = {z: np.where(skip_starts == z)[0] for z in lower_slices}
    skip_edge_ranges_to_slice  = {z: skip_ranges[skip_starts == z] for z in lower_slices}

    # we only need to calculate skip edge features for z edges
    filter_paths_z  = ds.make_filters(inp_ids[1], anisotropy_factor)

    skip_edge_features = np.zeros((skip_edges.shape[0], modified_features.shape[1]))
    for z in lower_slices:
        _get_skip_edge_features_for_slices(
            filter_paths_z,
            z,
            seg,
            skip_edge_pairs_to_slice[z],
            skip_edge_ranges_to_slice[z],
            skip_edge_indices_to_slice[z],
            skip_edge_features,
            z_direction=z_direction
        )

    skip_edge_features = np.nan_to_num(skip_edge_features)
    assert skip_edge_features.shape[1] == modified_features.shape[1]
    return np.nan_to_num(np.concatenate([modified_features, skip_edge_features], axis=0))


def _get_skip_edge_features_for_slices(
        filter_paths,
        z_dn,
        seg,
        skip_edge_pairs,
        skip_edge_ranges,
        skip_edge_indices,
        skip_edge_features,
        z_direction=0
):

    unique_ranges = np.unique(skip_edge_ranges)
    targets = unique_ranges + z_dn

    n_threads = ExperimentSettings().n_threads
    print "Computing skip edge features from slice ", z_dn
    for i, z_up in enumerate(targets):
        print "to", z_up

        which_skip_edges = skip_edge_ranges == unique_ranges[i]
        skip_pairs_z   = skip_edge_pairs[which_skip_edges]
        assert skip_pairs_z.shape[1] == 2
        skip_indices_z = skip_edge_indices[which_skip_edges]

        seg_local = np.concatenate(
            [seg[z_dn, :][None, :], seg[z_up, :][None, :]],
            axis=0
        )
        rag_local = nrag.gridRag(seg_local, n_threads)
        target_features = []
        for path in filter_paths:
            with h5py.File(path) as f:
                filt_ds = f['data']
                filt = np.concatenate(
                    [filt_ds[z_dn, :][None, :], filt_ds[z_up, :][None, :]],
                    axis=0
                )
            if len(filt.shape) == 3:
                target_features.append(
                    nrag.accumulateEdgeFeaturesFlat(rag_local, filt, filt.min(), filt.max(), z_direction, n_threads))
            elif len(filt.shape) == 4:
                for c in range(filt.shape[3]):
                    filt_c = filt[..., c]
                    target_features.append(
                        nrag.accumulateEdgeFeaturesFlat(rag_local, filt_c, filt_c.min(), filt_c.max(), 0, n_threads))

        target_features = np.concatenate(target_features, axis=1)
        # keep only the features corresponding to skip edges
        uvs_local = np.sort(rag_local.uvIds(), axis=1)
        assert uvs_local.shape[0] == target_features.shape[0]
        assert uvs_local.shape[1] == skip_edge_pairs.shape[1]

        # find the uvs_local that match skip edges
        matches = find_matching_row_indices(uvs_local, skip_pairs_z)
        # make sure that all skip edges were found
        assert matches.shape[0] == skip_pairs_z.shape[0], "%s, %s" % (str(matches.shape), str(skip_pairs_z.shape))
        # get the target features corresponding to skip edges and order them correctly
        target_features = target_features[matches[:, 0]][matches[:, 1]]

        # write the features to the feature array
        skip_edge_features[skip_indices_z, :] = target_features


@cacher_hdf5(folder="feature_folder", cache_edgefeats=True)
def modified_edge_features(ds, seg_id, inp_id, anisotropy_factor):

    modified_features = ds.edge_features(seg_id, inp_id, anisotropy_factor)

    skip_edges   = get_skip_edges( ds, seg_id)
    skip_starts  = get_skip_starts(ds, seg_id)
    skip_ranges  = get_skip_ranges(ds, seg_id)
    delete_edge_ids = get_delete_edge_ids(ds, seg_id)

    if not ds.has_defects:
        return modified_features

    # delete features for delete edges
    modified_features = np.delete(modified_features, delete_edge_ids, axis=0)

    # get features for skip edges
    seg = ds.seg(seg_id)
    lower_slices  = np.unique(skip_starts)
    skip_edge_pairs_to_slice   = {z: skip_edges[skip_starts == z] for z in lower_slices}
    skip_edge_indices_to_slice = {z: np.where(skip_starts == z)[0] for z in lower_slices}
    skip_edge_ranges_to_slice  = {z: skip_ranges[skip_starts == z] for z in lower_slices}

    filter_paths = ds.make_filters(inp_id, anisotropy_factor)

    skip_edge_features = np.zeros((skip_edges.shape[0], modified_features.shape[1]))
    for z in lower_slices:
        _get_skip_edge_features_for_slices(
            filter_paths,
            z,
            seg,
            skip_edge_pairs_to_slice[z],
            skip_edge_ranges_to_slice[z],
            skip_edge_indices_to_slice[z],
            skip_edge_features
        )

    skip_edge_features = np.nan_to_num(skip_edge_features)
    assert skip_edge_features.shape[1] == modified_features.shape[1]
    return np.concatenate([modified_features, skip_edge_features], axis=0)


@cacher_hdf5(folder="feature_folder", ignoreNumpyArrays=True)
def modified_region_features(ds, seg_id, inp_id, uv_ids, lifted_nh):

    modified_features = ds.region_features(seg_id, inp_id, uv_ids, lifted_nh)

    skip_edges   = get_skip_edges( ds, seg_id)
    skip_ranges  = get_skip_ranges(ds, seg_id)
    delete_edge_ids = get_delete_edge_ids(ds, seg_id)

    if not ds.has_defects:
        modified_features = np.c_[
            modified_features,
            np.logical_not(ds.edge_indications(seg_id)).astype('float32')
        ]
        return modified_features

    # delete all features corresponding to delete - edges
    modified_features = np.delete(modified_features, delete_edge_ids, axis=0)
    modified_features = np.c_[modified_features, np.ones(modified_features.shape[0])]

    ds._region_statistics(seg_id, inp_id)
    region_statistics_path = cache_name("_region_statistics", "feature_folder", False, False, ds, seg_id, inp_id)
    # add features for the skip edges
    region_stats = vigra.readHDF5(region_statistics_path, 'region_statistics')

    fU = region_stats[skip_edges[:, 0], :]
    fV = region_stats[skip_edges[:, 1], :]

    skip_stat_feats = np.concatenate([
        np.minimum(fU, fV),
        np.maximum(fU, fV),
        np.abs(fU - fV),
        fU + fV],
        axis=1
    )

    # features based on region center differences
    region_centers = vigra.readHDF5(region_statistics_path, 'region_centers')
    sU = region_centers[skip_edges[:, 0], :]
    sV = region_centers[skip_edges[:, 1], :]
    skip_center_feats = np.c_[(sU - sV)**2, skip_ranges]

    assert skip_center_feats.shape[0] == skip_stat_feats.shape[0]
    skip_features = np.concatenate([skip_stat_feats, skip_center_feats], axis=1)
    assert skip_features.shape[1] == modified_features.shape[1], "%s, %s" % (str(skip_features.shape), str(modified_features.shape))

    return np.nan_to_num(np.concatenate([modified_features, skip_features], axis=0))


def _get_topo_feats(rag, seg, use_2d_edges):
    feats = []
    # length / area of the edge
    edge_lens = nrag.accumulateMeanAndLength(rag, np.zeros_like(seg, dtype='float32'))[0][:, 1:]
    feats.append(edge_lens)
    # extra feats for z-edges in 2,5 d
    if use_2d_edges:
        extra_feats, _ = _topology_features_impl(rag, seg, np.zeros(rag.edgeNum, dtype='uint8'), edge_lens)
        feats.extend(extra_feats)
    return np.concatenate(feats, axis=1)


def _get_skip_topo_features_for_slices(
        z_dn,
        seg,
        skip_edge_pairs,
        skip_edge_ranges,
        skip_edge_indices,
        use_2d_edges,
        skip_edge_features
):

    unique_ranges = np.unique(skip_edge_ranges)
    targets = unique_ranges + z_dn

    print "Computing skip edge features from slice ", z_dn
    for i, z_up in enumerate(targets):
        print "to", z_up

        which_skip_edges = skip_edge_ranges == unique_ranges[i]
        skip_pairs_z   = skip_edge_pairs[which_skip_edges]
        assert skip_pairs_z.shape[1] == 2
        skip_indices_z = skip_edge_indices[which_skip_edges]

        seg_local = np.concatenate(
            [seg[z_dn, :][None, :], seg[z_up, :][None, :]],
            axis=0
        )
        rag_local = nrag.gridRag(seg, ExperimentSettings().n_threads)
        topo_feats = _get_topo_feats(rag_local, seg_local, use_2d_edges)

        # keep only the features corresponding to skip edges
        uvs_local = rag_local.uvIds()
        assert uvs_local.shape[0] == topo_feats.shape[0]
        assert uvs_local.shape[1] == skip_edge_pairs.shape[1]
        # find the uvs_local that match skip edges
        matches = find_matching_row_indices(uvs_local, skip_pairs_z)
        # make sure that all skip edges were found
        assert matches.shape[0] == skip_pairs_z.shape[0], "%s, %s" % (str(matches.shape), str(skip_edge_pairs.shape))
        # get the target features corresponding to skip edges and order them correctly
        topo_feats = topo_feats[matches[:, 0]][matches[:, 1]]
        # write the features to the feature array
        skip_edge_features[skip_indices_z, :] = topo_feats


@cacher_hdf5(folder="feature_folder")
def modified_topology_features(ds, seg_id, use_2d_edges):

    modified_features = ds.topology_features(seg_id, use_2d_edges)

    skip_edges   = get_skip_edges( ds, seg_id)
    skip_ranges  = get_skip_ranges(ds, seg_id)
    skip_starts  = get_skip_starts(ds, seg_id)
    delete_edge_ids = get_delete_edge_ids(ds, seg_id)

    if not ds.has_defects:
        return modified_features

    # delete all features corresponding to delete - edges
    modified_features = np.delete(modified_features, delete_edge_ids, axis=0)

    # get topo features for the new skip edges
    seg = ds.seg(seg_id)
    lower_slices  = np.unique(skip_starts)
    skip_edge_pairs_to_slice = {z: skip_edges[skip_starts == z] for z in lower_slices}
    skip_edge_indices_to_slice = {z: np.where(skip_starts == z)[0] for z in lower_slices}
    skip_edge_ranges_to_slice  = {z: skip_ranges[skip_starts == z] for z in lower_slices}

    n_feats = modified_features.shape[1]
    skip_topo_features = np.zeros((skip_edges.shape[0], n_feats))

    for z in lower_slices:
        _get_skip_topo_features_for_slices(
            z,
            seg,
            skip_edge_pairs_to_slice[z],
            skip_edge_ranges_to_slice[z],
            skip_edge_indices_to_slice[z],
            use_2d_edges,
            skip_topo_features
        )

    skip_topo_features[np.isinf(skip_topo_features)] = 0.
    skip_topo_features[np.isneginf(skip_topo_features)] = 0.
    skip_topo_features = np.nan_to_num(skip_topo_features)
    assert skip_topo_features.shape[1] == modified_features.shape[1]
    return np.concatenate([modified_features, skip_topo_features], axis=0)


# the last argument is only for caching results with different features correctly
@cacher_hdf5(ignoreNumpyArrays=True)
def modified_probs_to_energies(
        ds,
        edge_probs,
        seg_id,
        uv_ids,
        weighting_scheme,
        weight,
        beta,
        feat_cache):

    assert edge_probs.shape[0] == uv_ids.shape[0], "%s, %s" % (str(edge_probs.shape), str(uv_ids.shape))

    # scale the probabilities
    # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
    p_min = 0.001
    p_max = 1. - p_min
    edge_probs = (p_max - p_min) * edge_probs + p_min

    # probabilities to energies, second term is boundary bias
    edge_energies = np.log((1. - edge_probs) / edge_probs) + np.log((1. - beta) / beta)

    if weighting_scheme in ("z", "xyz", "all"):
        edge_areas       = modified_topology_features(ds, seg_id, False)[:, 0]
        edge_indications = modified_edge_indications(ds, seg_id)

    # weight edges
    if weighting_scheme == "z":
        print "Weighting Z edges"
        edge_energies = weight_z_edges(edge_energies, edge_areas, edge_indications, weight)
    elif weighting_scheme == "xyz":
        print "Weighting xyz edges"
        edge_energies = weight_xyz_edges(edge_energies, edge_areas, edge_indications, weight)
    elif weighting_scheme == "all":
        print "Weighting all edges"
        edge_energies = weight_all_edges(edge_energies, edge_areas, weight)

    # set ignore edges to be maximally repulsive
    ignore_edge_ids = get_ignore_edge_ids(ds, seg_id)
    max_repulsive = 2 * edge_energies.min()
    if ignore_edge_ids.size:
        edge_energies[ignore_edge_ids] = max_repulsive

    # set the edges within the segmask to be maximally repulsive
    if ds.has_seg_mask:
        ignore_mask = (uv_ids == ExperimentSettings().ignore_seg_value).any(axis=1)
        edge_energies[ignore_mask] = max_repulsive

    assert not np.isnan(edge_energies).any()
    return edge_energies

#
# Segmentation Postprocessing
#


def _get_replace_slices(defected_slices, shape):
    # find consecutive slices with defects
    consecutive_defects = np.split(defected_slices, np.where(np.diff(defected_slices) != 1)[0] + 1)
    # find the replace slices for defected slices
    replace_slice = {}
    for consec in consecutive_defects:
        if len(consec) == 1:
            z = consec[0]
            replace_slice[z] = z - 1 if z > 0 else 1
            # replace_slice[z] = z + 1 if z > 0 else 1
        elif len(consec) == 2:
            z0, z1 = consec[0], consec[1]
            replace_slice[z0] = z0 - 1 if z0 > 0 else 2
            replace_slice[z1] = z1 + 1 if z1 < shape[0] - 1 else z1 - 2
        elif len(consec) == 3:
            z0, z1, z2 = consec[0], consec[1], consec[2]
            replace_slice[z0] = z0 - 1 if z0 > 0 else 3
            replace_slice[z1] = z1 - 2 if z1 > 1 else 3
            replace_slice[z2] = z2 + 1 if z2 < shape[0] - 1 else z2 - 3
        elif len(consec) == 3:
            z0, z1, z2, z3 = consec[0], consec[1], consec[2], consec[3]
            replace_slice[z0] = z0 - 1 if z0 > 0 else 4
            replace_slice[z1] = z1 - 2 if z1 > 1 else 4
            replace_slice[z2] = z2 + 1 if z2 < shape[0] - 1 else z2 - 3
            replace_slice[z3] = z3 + 2 if z3 < shape[0] - 1 else z3 - 4
        else:
            raise RuntimeError("Postprocessing is not implemented for more than 4 consecutively defected slices. Go and clean your data!")
    return replace_slice


def postprocess_segmentation(ds, seg_id, seg_result):
    if not ds.has_defects:
        return seg_result
    defect_slices = get_completely_defected_slices(ds, seg_id)
    assert defect_slices.size, "Can only post-process data with defects"
    replace_slices = _get_replace_slices(defect_slices, seg_result.shape)
    for defect_slice in defect_slices:
        replace = replace_slices[defect_slice]
        seg_result[:, :, defect_slice] = seg_result[:, :, replace]
    return seg_result


# FIXME I have now idea if this is still doing the right thing
def postprocess_segmentation_with_missing_slices(seg_result, slice_list):
    print "WARNING"
    print "WARNING: This function 'postprocess_segmentation_with_missing_slices' may do completely bogus things!"
    print "WARNING"
    replace_slices = _get_replace_slices(slice_list)
    total_insertions = 0
    for insrt in replace_slices:
        repl = replace_slices[insrt]
        repl += total_insertions
        slice_repl = seg_result[:, :, repl]
        np.insert(seg_result, slice_repl, axis=2)
        total_insertions += 1
    return seg_result
