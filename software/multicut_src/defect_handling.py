import vigra
import h5py
import numpy as np
from concurrent import futures
from Tools import cacher_hdf5, cache_name

from DataSet import DataSet, Cutout
from MCSolverImpl import weight_z_edges, weight_all_edges, weight_xyz_edges

#
# Defect detection
#

@cacher_hdf5()
def oversegmentation_statistics(ds, seg_id, n_bins):
    seg = ds.seg(seg_id)

    def extract_segs_in_slice(z):
        # 2d blocking representing the patches
        seg_z = seg[:,:,z]
        return np.unique(seg_z).shape[0]

    # parallel
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = []
        for z in xrange(seg.shape[2]):
            tasks.append(executor.submit(extract_segs_in_slice, z))
        segs_per_slice = [fut.result() for fut in tasks]

    # calculate histogram to have a closer look at the stats
    histo, bin_edges = np.histogram(segs_per_slice, bins = n_bins)
    # we only need the bin_edges
    return bin_edges

@cacher_hdf5()
def defect_slice_detection(ds, seg_id, n_bins, bin_threshold):

    bin_edges = oversegmentation_statistics(ds, seg_id, n_bins)
    seg = ds.seg(seg_id)

    threshold = bin_edges[bin_threshold]
    out = np.zeros_like(seg, dtype = 'uint8')

    def detect_defected_slice(z):
        seg_z = seg[:,:,z]
        # get number of segments for patches in this slice
        n_segs = np.unique(seg_z).shape[0]
        # threshold for a defected slice
        if n_segs < threshold:
            out[:,:,z] = 1
            return True
        else:
            return False

    with futures.ThreadPoolExecutor(max_workers = 8) as executor:
        tasks = []
        for z in xrange(seg.shape[2]):
            tasks.append(executor.submit(detect_defected_slice,z))
        defect_indications = [fut.result() for fut in tasks]

    # report the defects
    for z in xrange(seg.shape[2]):
        if defect_indications[z]:
            print "DefectSliceDetection: slice %i is defected." % z

    return out

#
# Modified Adjacency
#

@cacher_hdf5()
def defects_to_nodes(ds, seg_id, n_bins, bin_threshold):
    defects = defect_slice_detection(ds, seg_id, n_bins, bin_threshold)
    seg = ds.seg(seg_id)
    assert seg.shape == defects.shape

    def defects_to_nodes_z(z):
        defect_mask = defects[:,:,z]
        if 1 in defect_mask:
            seg_z = seg[:,:,z]
            where_defect = defect_mask == 1
            defect_nodes_slice = np.unique(seg_z[where_defect])
            return list(defect_nodes_slice), len(defect_nodes_slice) * [z]
        else:
            return [], []

    with futures.ThreadPoolExecutor(max_workers = 8) as executor:
        tasks = []
        for z in xrange(seg.shape[2]):
            tasks.append(executor.submit(defects_to_nodes_z,z))
        defect_nodes = []
        nodes_z      = []
        for fut in tasks:
            nodes, zz = fut.result()
            if nodes:
                defect_nodes.extend(nodes)
                nodes_z.extend(zz)

    assert len(defect_nodes) == len(nodes_z)

    # stupid caching... need to concatenate and later retrieve this...
    return np.concatenate([np.array(defect_nodes,dtype='uint32'), np.array(nodes_z,dtype='uint32')])

# this is very hacky due to stupid caching...
# we calculate everything with modified adjacency and then load the things with individual functions

def get_delete_edges(ds, seg_id, n_bins, bin_threshold):
    modified_adjacency(ds, seg_id, n_bins, bin_threshold)
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id, n_bins, bin_threshold)
    return vigra.readHDF5(mod_save_path, "delete_edges")

def get_ignore_edges(ds, seg_id, n_bins, bin_threshold):
    modified_adjacency(ds, seg_id, n_bins, bin_threshold)
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id, n_bins, bin_threshold)
    return vigra.readHDF5(mod_save_path, "ignore_edges")

def get_skip_edges(ds, seg_id, n_bins, bin_threshold):
    modified_adjacency(ds, seg_id, n_bins, bin_threshold)
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id, n_bins, bin_threshold)
    return vigra.readHDF5(mod_save_path, "skip_edges")

def get_skip_ranges(ds, seg_id, n_bins, bin_threshold):
    modified_adjacency(ds, seg_id, n_bins, bin_threshold)
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id, n_bins, bin_threshold)
    return vigra.readHDF5(mod_save_path, "skip_ranges")

def get_skip_starts(ds, seg_id, n_bins, bin_threshold):
    modified_adjacency(ds, seg_id, n_bins, bin_threshold)
    mod_save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id, n_bins, bin_threshold)
    return vigra.readHDF5(mod_save_path, "skip_starts")

# TODO TODO TODO debug
@cacher_hdf5()
def modified_adjacency(ds, seg_id, n_bins, bin_threshold):
    node_res = defects_to_nodes(ds, seg_id, n_bins, bin_threshold)
    # need to split into defect nodes and node_z
    mid = node_res.shape[0] / 2
    defect_nodes = node_res[:mid]
    nodes_z = node_res[mid:]

    # make sure that z is monotonically increasing (not strictly!)
    assert np.all(np.diff(nodes_z.astype(int)) >= 0), "Defected slice index is not increasing monotonically!"
    defect_slices = np.unique(nodes_z)
    defect_node_dict = {int(z) : list(defect_nodes[nodes_z == z].astype(int)) for z in defect_slices}

    # FIXME TODO can't do this here once we have individual defect patches
    consecutive_defect_slices = np.split(defect_slices, np.where(np.diff(defect_slices) != 1)[0] + 1)
    has_lower_defect_list = []
    for consec in consecutive_defect_slices:
        if len(consec) > 1:
            has_lower_defect_list.extend(consec[1:])

    # iterate over the nodes in slices with defects to get delete, ignore and skip edges
    rag = ds._rag(seg_id)
    seg = ds.seg(seg_id)
    edge_indications = ds.edge_indications(seg_id)

    def modified_adjacency_node(z_up, z_dn, nodes_dn, mask):
        skip_range = z_up - z_dn
        skip_edges, skip_ranges = [], []

        for node_dn in nodes_dn:
            # find the connected nodes in the upper slice
            coords_dn = np.where(seg[:,:,z_dn][mask] == node_dn)
            seg_up = seg[:,:,z_up][mask]
            connected_nodes = np.unique( seg_up[coords_dn] )
            # if any of the connected nodes are defected go to the next slice
            has_upper_defect = False
            for conn_node in connected_nodes:
                if conn_node in defect_node_dict.get(z_up,[]):
                    has_upper_defect = True
                    break
            if has_upper_defect:
                skip_edges, skip_ranges = modified_adjacency_node(z_up+1, z_dn, nodes_dn, mask)
                break
            else:
                for conn_node in connected_nodes:
                    skip_edges.append((node_dn, conn_node))
                    skip_ranges.append(skip_range)
        return skip_edges, skip_ranges

    # FIXME this won't really parallelize due to GIL
    def modified_adjacency_z(z, has_lower_defect):
        defect_nodes_z = defect_node_dict[z]
        delete_edges_z = []
        ignore_edges_z = []

        # get delete and ignore edges in slice
        for defect_node in defect_nodes_z:
            rag_node = rag.nodeFromId(defect_node)
            for nn_node in rag.neighbourNodeIter(rag_node):
                edge_id = rag.findEdge(rag_node, nn_node).id
                if edge_indications[edge_id]: # we have a in-plane edge -> add this to the ignore edges, if the neighbouring node is also defected
                    if nn_node.id in defect_nodes_z:
                        # we store the uv-ids for ignore edges, because the actual edge id will change due to skip and delete edges
                        ignore_edges_z.append( (rag_node.id, nn_node.id) )
                else: # we have a in-between-planes edge -> add this to the delete edges
                    delete_edges_z.append(edge_id)

        # get the skip edges between adjacent slices
        # skip for first or last slice or slice with lower defect
        if z == 0 or z == seg.shape[2] - 1 or has_lower_defect:
            return delete_edges_z, ignore_edges_z, [], []

        skip_edges_z = []
        skip_ranges_z = []

        mask = np.zeros(seg.shape[:2], dtype = bool)
        coords_u_prev = []

        for defect_node in defect_nodes_z:
            # reset the mask
            if coords_u_prev:
                mask[coords_u_prev] = False
            # get the coords of this node
            coords_u = np.where(seg[:,:,z] == defect_node)
            # set the mask
            mask[coords_u] = True
            # find the lower nodes overlapping with the defect in the lower slice
            nodes_dn = np.unique( seg[:,:,z-1][mask] )
            # discard defected nodes in lower slice (if present) because they were already taken care of
            nodes_dn = np.array([n_dn for n_dn in nodes_dn if n_dn not in defect_nodes_z])
            # if we have lower nodes left, we look for skip edges
            if nodes_dn.size:
                skip_edges_u, skip_ranges_u = modified_adjacency_node(z+1, z-1, nodes_dn, mask)
                skip_edges_z.extend(skip_edges_u)
                skip_ranges_z.extend(skip_ranges_u)
        return delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z

    delete_edges = [] # the z-edges between defected and non-defected nodes that are deleted from the graph
    ignore_edges = [] # the xy-edges between defected and non-defected nodes, that will be set to maximally repulsive weights

    skip_edges   = [] # the skip edges that run over the defects in z
    skip_ranges  = [] # z-distance of the skip edges
    skip_starts  = [] # starting slices of the skip edges

    # sequential for now
    #with futures.ThreadPoolExecutor(max_workers = 8) as executor:
    #    tasks = []
    for i,z in enumerate(defect_slices):
        print "Processing slice %i: %i / %i" % (z,i,len(defect_slices))
        has_lower_defect = True if z in has_lower_defect_list else False
        delete_edges_z, ignore_edges_z, skip_edges_z, skip_ranges_z = modified_adjacency_z( z, has_lower_defect)
        delete_edges.extend(delete_edges_z)
        ignore_edges.extend(ignore_edges_z)
        assert len(skip_edges_z) == len(skip_ranges_z)
        skip_edges.extend(skip_edges_z)
        skip_ranges.extend(skip_ranges_z)
        skip_starts.extend(len(skip_edges_z) * [z-1])

    delete_edges = np.unique(delete_edges).astype('uint32')
    ignore_edges = np.array(ignore_edges).astype('uint32')

    skip_edges = np.array(skip_edges, dtype = np.uint32)
    # make the skip edges unique, keeping rows (see http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array):
    skips_view = np.ascontiguousarray(skip_edges).view(np.dtype((np.void, skip_edges.dtype.itemsize * skip_edges.shape[1])))
    _, idx = np.unique(skips_view, return_index=True)
    skip_edges = skip_edges[idx]

    skip_ranges = np.array(skip_ranges, dtype = np.uint32)[idx]
    skip_starts = np.array(skip_starts, dtype = np.uint32)[idx]
    assert skip_edges.shape[0] == skip_ranges.shape[0]
    assert skip_starts.shape[0] == skip_ranges.shape[0]

    # reorder the skip edges s.t. skip_starts are monotonically increasing
    sort_indices = np.argsort(skip_starts)
    skip_edges = skip_edges[sort_indices]
    skip_ranges = skip_ranges[sort_indices]
    skip_starts = skip_starts[sort_indices]
    # make sure that z is monotonically increasing (not strictly!)
    assert np.all(np.diff(skip_starts.astype(int)) >= 0), "start index of skip edges must increase monotonically."

    # save delete, ignore and skip edges, a little hacky due to stupid caching...
    save_path = cache_name("modified_adjacency", "dset_folder", False, False, ds, seg_id, n_bins, bin_threshold)
    vigra.writeHDF5(delete_edges,save_path, "delete_edges")
    vigra.writeHDF5(ignore_edges,save_path, "ignore_edges")
    vigra.writeHDF5(skip_edges,  save_path, "skip_edges")
    vigra.writeHDF5(skip_ranges, save_path, "skip_ranges")
    vigra.writeHDF5(skip_starts, save_path, "skip_starts")
    return []


@cacher_hdf5()
def modified_edge_indications(ds, seg_id, n_bins, bin_threshold):
    modified_indications = ds.edge_indications(seg_id)
    skip_edges   = get_skip_edges(ds, seg_id, n_bins, bin_threshold)
    delete_edges = get_delete_edges(ds, seg_id, n_bins, bin_threshold)
    modified_indications = np.delete(modified_indications, delete_edges)
    return np.concatenate( [modified_indications, np.zeros(skip_edges.shape[0], dtype = modified_indications.dtype)] )


@cacher_hdf5()
def modified_edge_gt(ds, seg_id, n_bins, bin_threshold):
    modified_edge_gt = ds.edge_gt(seg_id)
    skip_edges   = get_skip_edges(ds, seg_id, n_bins, bin_threshold)
    delete_edges = get_delete_edges(ds, seg_id, n_bins, bin_threshold)
    modified_edge_gt = np.delete(modified_edge_gt, delete_edges)
    rag = ds._rag(seg_id)
    node_gt, _ = rag.projectBaseGraphGt( ds.gt().astype('uint32') )
    skip_gt = (node_gt[skip_edges[:,0]] != node_gt[skip_edges[:,1]]).astype('float32')
    return np.concatenate([modified_edge_gt, skip_gt])


#
# Modified Features
#

# TODO modified edge features from affinities -> implement!!!
def modified_edge_features_from_affinity_maps(ds, seg_id, inp_id, anisotropy_factor, n_bins, bin_threshold):
    pass

def _get_skip_edge_features_for_slices(filter_paths, z_dn,
        targets, seg,
        skip_edge_pairs, skip_edge_indices,
        skip_edge_features):

    features = []
    for z_up in targets:
        seg_local = np.concatenate([seg[:,:,z_dn][:,:,None],seg[:,:,z_up][:,:,None]],axis=2)
        rag_local = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(seg_local.shape),seg_local)
        target_features = []
        for path in filter_paths:
            with h5py.File(path) as f:
                filt_ds = f['data']
                filt = np.concatenate([filt_ds[:,:,z_dn][:,:,None],filt_ds[:,:,z_up][:,:,None]],axis=2)
            if len(filt.shape) == 3:
                gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(rag_local.baseGraph, filt)
                edgeFeats     = rag_local.accumulateEdgeStatistics(gridGraphEdgeIndicator)
                target_features.append(edgeFeats)
            elif len(filt.shape) == 4:
                for c in range(filt.shape[3]):
                    gridGraphEdgeIndicator = vigra.graphs.implicitMeanEdgeMap(
                            rag_local.baseGraph, filt[:,:,:,c] )
                    edgeFeats     = rag_local.accumulateEdgeStatistics(gridGraphEdgeIndicator)
                    target_features.append(edgeFeats)
        target_features = np.concatenate(target_features, axis = 1)
        # keep only the features corresponding to skip edges
        uvs_local = np.sort(rag_local.uvIds(), axis = 1)
        assert uvs_local.shape[0] == target_features.shape[0]
        assert uvs_local.shape[1] == skip_edge_pairs.shape[1]
        # FIXME horrible loop....
        keep_indices = []
        to_skip_edges = np.zeros(skip_edge_pairs.shape[0], dtype = 'uint32')
        found = 0
        for i, uv in enumerate(uvs_local):
            where_uv = np.where(np.all(uv == skip_edge_pairs, axis = 1))[0]
            if where_uv.size:
                assert where_uv[0].size == 1
                keep_indices.append(i)
                to_skip_edges[where_uv[0]] = found
                found += 1
        assert len(keep_indices) == skip_edge_pairs.shape[0]
        assert len(to_skip_edges) == skip_edge_pairs.shape[0]
        features.append(target_features[keep_indices][to_skip_edges])
    features = np.concatenate(features, axis = 0)
    skip_edge_features[skip_edge_indices,:] = features


@cacher_hdf5(folder="feature_folder", cache_edgefeats=True)
def modified_edge_features(ds, seg_id, inp_id, anisotropy_factor, n_bins, bin_threshold):
    modified_features = ds.edge_features(seg_id, inp_id, anisotropy_factor)

    skip_edges   = get_skip_edges(ds, seg_id, n_bins, bin_threshold)
    skip_starts  = get_skip_starts(ds, seg_id, n_bins, bin_threshold)
    skip_ranges  = get_skip_ranges(ds, seg_id, n_bins, bin_threshold)
    delete_edges = get_delete_edges(ds, seg_id, n_bins, bin_threshold)

    # delete features for delete edges
    modified_features = np.delete(modified_features, delete_edges, axis = 0)

    # get features for skip edges
    seg = ds.seg(seg_id)
    lower_slices  = np.unique(skip_starts)
    skip_edge_pairs_to_slice = {z : skip_edges[skip_starts == z] for z in lower_slices}
    skip_edge_indices_to_slice = {z : np.where(skip_starts == z)[0] for z in lower_slices}
    target_slices = {z : z + np.unique(skip_ranges[skip_starts == z]) for z in lower_slices}

    # calculate the volume filters for the given input
    if isinstance(ds, Cutout):
        filter_paths = ds.make_filters(inp_id, anisotropy_factor, ds.ancestor_folder)
    else:
        filter_paths = ds.make_filters(inp_id, anisotropy_factor)

    skip_edge_features = np.zeros( (skip_edges.shape[0], modified_features.shape[1]) )
    for z in lower_slices:
        this_skip_edge_pairs = skip_edge_pairs_to_slice[z]
        this_skip_edge_indices = skip_edge_indices_to_slice[z]
        target = target_slices[z]
        _get_skip_edge_features_for_slices(filter_paths,
                z, target,
                seg, this_skip_edge_pairs,
                this_skip_edge_indices, skip_edge_features)

    skip_edge_features = np.nan_to_num(skip_edge_features)
    assert skip_edge_features.shape[1] == modified_features.shape[1]
    return np.concatenate([modified_features, skip_edge_features],axis = 0)


@cacher_hdf5(folder="feature_folder", ignoreNumpyArrays=True)
def modified_region_features(ds, seg_id, inp_id, uv_ids, lifted_nh, n_bins, bin_threshold):
    modified_features = ds.region_features(seg_id, inp_id, uv_ids, lifted_nh)

    skip_edges   = get_skip_edges(ds, seg_id, n_bins, bin_threshold)
    skip_ranges  = get_skip_ranges(ds, seg_id, n_bins, bin_threshold)
    delete_edges = get_delete_edges(ds, seg_id, n_bins, bin_threshold)

    # delete all features corresponding to delete - edges
    modified_features = np.delete(modified_features, delete_edges, axis = 0)
    modified_features = np.c_[modified_features, np.ones(modified_features.shape[0])]

    # add features for the skip edges
    extracted_features, stat_names  = ds._region_statistics(seg_id, inp_id)
    node_features = np.concatenate(
        [extracted_features[stat_name][:,None] if extracted_features[stat_name].ndim == 1 else extracted_features[stat_name] for stat_name in stat_names],
        axis = 1)

    #del extracted_features

    print node_features.shape
    n_stat_feats = 17 # magic_nu...
    region_stats = node_features[:,:n_stat_feats]

    fU = region_stats[skip_edges[:,0],:]
    fV = region_stats[skip_edges[:,1],:]

    skip_stat_feats = np.concatenate([np.minimum(fU,fV),
        np.maximum(fU,fV),
        np.abs(fU - fV),
        fU + fV], axis = 1)

    # features based on region center differences
    region_centers = node_features[:,n_stat_feats:]
    sU = region_centers[skip_edges[:,0],:]
    sV = region_centers[skip_edges[:,1],:]
    skip_center_feats = np.c_[(sU - sV)**2, skip_ranges]

    assert skip_center_feats.shape[0] == skip_stat_feats.shape[0]
    skip_features = np.concatenate([skip_stat_feats, skip_center_feats], axis = 1)
    assert skip_features.shape[1] == modified_features.shape[1], "%s, %s" % (str(skip_features.shape), str(modified_features.shape))

    return np.concatenate( [modified_features, skip_features], axis = 0)


def _get_skip_topo_features_for_slices(z_dn, targets,
        seg, skip_edge_pairs,
        skip_edge_indices, use_2d_edges,
        skip_edge_features):

    features = []
    for z_up in targets:
        seg_local = np.concatenate([seg[:,:,z_dn][:,:,None],seg[:,:,z_up][:,:,None]],axis=2)
        rag_local = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(seg_local.shape),seg_local)
        target_features = []

        # length / area of the edge
        edge_lens = rag_local.edgeLengths()
        target_features.append(edge_lens[:,None])

        # extra feats for z-edges in 2,5 d
        if use_2d_edges:
            # edge indications -> these are 0 (=z-edge) for all skip edges
            target_features.append(np.zeros(rag_local.edgeNum)[:,None])
            # region sizes to build some features
            statistics =  [ "Count", "RegionCenter" ]
            extractor = vigra.analysis.extractRegionFeatures(
                    np.zeros_like(seg, dtype = 'float32'),
                    seg.astype(np.uint32),
                    features = statistics )

            sizes = extractor["Count"]
            uvIds = rag_local.uvIds()
            sizes_u = sizes[ uvIds[:,0] ]
            sizes_v = sizes[ uvIds[:,1] ]

            unions  = sizes_u + sizes_v - edge_lens
            # Union features
            target_features.append( unions[:,None] )
            # IoU features
            target_features.append( (edge_lens / unions)[:,None] )

            # segment shape features
            seg_coordinates = extractor["RegionCenter"]
            len_bounds      = {n.id : 0. for n in rag_local.nodeIter()}
            # TODO no loop ?! or CPP
            # iterate over the nodes, to get the boundary length of each node
            for n in rag_local.nodeIter():
                node_z = seg_coordinates[n.id][2]
                for arc in rag_local.incEdgeIter(n):
                    edge = rag_local.edgeFromArc(arc)
                    edge_c = rag_local.edgeCoordinates(edge)
                    # only edges in the same slice!
                    if edge_c[0,2] == node_z:
                        len_bounds[n.id] += edge_lens[edge.id]
            # shape feature = Area / Circumference
            shape_feats_u = sizes_u / np.array( [ len_bounds[u] for u in uvIds[:,0] ] )
            shape_feats_v = sizes_v / np.array( [ len_bounds[v] for v in uvIds[:,1] ] )
            # combine w/ min, max, absdiff
            target_features.append( np.minimum( shape_feats_u, shape_feats_v)[:,None] )
            target_features.append( np.maximum( shape_feats_u, shape_feats_v)[:,None])
            target_features.append( np.absolute(shape_feats_u - shape_feats_v)[:,None] )

        target_features = np.concatenate(target_features, axis = 1)
        # keep only the features corresponding to skip edges
        uvs_local = np.sort(rag_local.uvIds(), axis = 1)
        assert uvs_local.shape[0] == target_features.shape[0]
        assert uvs_local.shape[1] == skip_edge_pairs.shape[1]
        # FIXME horrible loop....
        keep_indices = []
        to_skip_edges = np.zeros(skip_edge_pairs.shape[0], dtype = 'uint32')
        found = 0
        for i, uv in enumerate(uvs_local):
            where_uv = np.where(np.all(uv == skip_edge_pairs, axis = 1))[0]
            if where_uv.size:
                assert where_uv[0].size == 1
                keep_indices.append(i)
                to_skip_edges[where_uv[0]] = found
                found += 1
        assert len(keep_indices) == skip_edge_pairs.shape[0]
        assert len(to_skip_edges) == skip_edge_pairs.shape[0]
        features.append(target_features[keep_indices][to_skip_edges])
    features = np.concatenate(features, axis = 0)
    skip_edge_features[skip_edge_indices,:] = features


@cacher_hdf5(folder="feature_folder")
def modified_topology_features(ds, seg_id, use_2d_edges, n_bins, bin_threshold):
    modified_features = ds.topology_features(seg_id, use_2d_edges)

    skip_edges   = get_skip_edges(ds, seg_id, n_bins, bin_threshold)
    skip_ranges  = get_skip_ranges(ds, seg_id, n_bins, bin_threshold)
    skip_starts  = get_skip_starts(ds, seg_id, n_bins, bin_threshold)
    delete_edges = get_delete_edges(ds, seg_id, n_bins, bin_threshold)

    # delete all features corresponding to delete - edges
    modified_features = np.delete(modified_features, delete_edges, axis = 0)

    # get topo features for the new skip edges
    seg = ds.seg(seg_id)
    lower_slices  = np.unique(skip_starts)
    skip_edge_pairs_to_slice = {z : skip_edges[skip_starts == z] for z in lower_slices}
    skip_edge_indices_to_slice = {z : np.where(skip_starts == z) for z in lower_slices}
    target_slices = {z : z + np.unique(skip_ranges[skip_starts == z]) for z in lower_slices}

    n_feats = modified_features.shape[1]
    skip_topo_features = np.zeros( (skip_edges.shape[0], n_feats) )

    for z in lower_slices:
        this_skip_edge_pairs = skip_edge_pairs_to_slice[z]
        this_skip_edge_indices = skip_edge_indices_to_slice[z]
        target = target_slices[z]
        _get_skip_topo_features_for_slices(z, target,
                seg, this_skip_edge_pairs,
                this_skip_edge_indices, use_2d_edges,
                skip_topo_features)

    skip_topo_features[np.isinf(skip_topo_features)] = 0.
    skip_topo_features[np.isneginf(skip_topo_features)] = 0.
    skip_topo_features = np.nan_to_num(skip_topo_features)
    assert skip_topo_features.shape[1] == modified_features.shape[1]
    return np.concatenate([modified_features, skip_topo_features],axis = 0)

#
# Modified Multicut Problem
#

def modified_mc_problem(ds, seg_id, n_bins, bin_threshold):
    modified_uv_ids = ds._adjacent_segments(seg_id)
    skip_edges   = np.sort( get_skip_edges(ds, seg_id, n_bins, bin_threshold), axis = 1)
    delete_edges = get_delete_edges(ds, seg_id, n_bins, bin_threshold)
    modified_uv_ids = np.delete(modified_uv_ids, delete_edges, axis = 0)
    modified_uv_ids = np.concatenate([modified_uv_ids, skip_edges])
    assert modified_uv_ids.shape[1] == 2, str(modified_uv_ids.shape)
    n_var = ds.seg(seg_id).max() + 1
    return n_var, modified_uv_ids


# the last argument is only for caching results with different features correctly
@cacher_hdf5(ignoreNumpyArrays=True)
def modified_probs_to_energies(ds, edge_probs, seg_id, uv_ids, exp_params, n_bins, bin_threshold, feat_cache):

    # scale the probabilities
    # this is pretty arbitrary, it used to be 1. / n_tress, but this does not make that much sense for sklearn impl
    p_min = 0.001
    p_max = 1. - p_min
    edge_probs = (p_max - p_min) * edge_probs + p_min

    # probabilities to energies, second term is boundary bias
    edge_energies = np.log( (1. - edge_probs) / edge_probs ) + np.log( (1. - exp_params.beta_local) / exp_params.beta_local )

    if exp_params.weighting_scheme in ("z", "xyz", "all"):
        edge_areas       = modified_topology_features(ds, seg_id, False, n_bins, bin_threshold)[:,0]
        edge_indications = modified_edge_indications(ds, seg_id, n_bins, bin_threshold)

    # weight edges
    if exp_params.weighting_scheme == "z":
        print "Weighting Z edges"
        edge_energies = weight_z_edges(ds, edge_energies, seg_id, edge_areas, edge_indications, exp_params.weight)
    elif exp_params.weighting_scheme == "xyz":
        print "Weighting xyz edges"
        edge_energies = weight_xyz_edges(ds, edge_energies, seg_id, edge_areas, edge_indications, exp_params.weight)
    elif exp_params.weighting_scheme == "all":
        print "Weighting all edges"
        edge_energies = weight_all_edges(ds, edge_energies, seg_id, edge_areas, exp_params.weight)

    # set ignore edges to be maximally repulsive
    ignore_edges = get_ignore_edges(ds, seg_id, n_bins, bin_threshold)
    if ignore_edges.size:
        assert uv_ids.shape[1] == ignore_edges.shape[1]
        # FIXME horrible loop....
        ignore_indices = []
        for uv in uv_ids:
            where_uv = np.where(np.all(uv == ignore_edges, axis = 1))
            if where_uv[0].size:
                assert where_uv[0].size == 1
                ignore_indices.append(where_uv[0][0])
        ignore_indices = np.sort(ignore_indices)
        max_repulsive = 2 * edge_energies.min()
        edge_energies[ignore_indices] = max_repulsive

    assert not np.isnan(edge_energies).any()
    return edge_energies

#
# Segmentation Postprocessing
#

# TODO modified features, need to figure out how to do this exactly ...
def _get_replace_slices(defected_slices, shape):
    # find consecutive slices with defects
    consecutive_defects = np.split(defected_slices, np.where(np.diff(defected_slices) != 1)[0] + 1)
    # find the replace slices for defected slices
    replace_slice = {}
    for consec in consecutive_defects:
        if len(consec) == 1:
            z = consec[0]
            replace_slice[z] = z - 1 if z > 0 else 1
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


def postprocess_segmentation(ds, seg_id, seg_result, n_bins, bin_threshold):
    defect_nodes = defects_to_nodes(ds, seg_id, n_bins, bin_threshold)
    mid = defect_nodes.shape[0] / 2
    defect_slices = np.unique(defect_nodes[mid:])
    replace_slices = _get_replace_slices(defect_slices, seg_result.shape)
    for defect_slice in defect_slices:
        replace = replace_slices[defect_slice]
        seg_result[:,:,defect_slice] = seg_result[:,:,replace]
    return seg_result

def postprocess_segmentation_with_missing_slices(seg_result, slice_list):
    replace_slices = _get_replace_slices(slice_list)
    total_insertions = 0
    for insrt in replace_slices:
        repl = replace_slices[insrt_slice]
        insrt += total_insertions
        repl += total_insertions
        slice_repl = seg_result[:,:,repl]
        np.insert( seg_result, slice_repl, axis = 2)
        total_insertions += 1
    return seg_result
