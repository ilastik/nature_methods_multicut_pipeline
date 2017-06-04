import numpy as np
import vigra

from tools import find_matching_row_indices, replace_from_dict, find_exclusive_matching_indices

# from concurrent import futures
# from ExperimentSettings import ExperimentSettings

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty.cgp as ncgp
    import nifty.graph.rag as nrag
except ImportError:
    try:
        import nifty_with_cplex.cgp as ncgp
        import nifty_with_cplex.graph.rag as nrag
    except ImportError:
        try:
            import nifty_with_gurobi.cgp as ncgp
            import nifty_with_gurobi.graph.rag as nrag
        except ImportError:
            raise ImportError("No valid nifty version was found.")


def topo_feats_slice(seg, uv_ids):

    # get the segmentation in this slice and map it
    # to a consecutive labeling starting from 1
    seg_z, _, mapping = vigra.analysis.relabelConsecutive(
        seg,
        keep_zeros=False,
        start_label=1
    )
    reverse_mapping = {new: old for old, new in mapping.iteritems()}
    assert seg_z.min() == 1

    tgrid = ncgp.TopologicalGrid2D(seg_z)

    # seg_cc = nseg.connectedComponents(seg_z)
    # uniques = np.unique(seg)
    # uniques_cc = np.unique(seg_cc)
    # print len(uniques), len(uniques_cc)
    # assert len(uniques) == len(uniques_cc), "%i, %i" % (len(uniques), len(uniques_cc))

    # extract the cell geometry
    cell_geometry = tgrid.extractCellsGeometry()
    cell_bounds = tgrid.extractCellsBounds()

    # get curvature feats, needs cell1geometry vector and cell1boundedby vector
    curvature_calculator = ncgp.Cell1CurvatureFeatures2D()
    curve_feats_ = curvature_calculator(
        cell_geometry[1],
        cell_bounds[0].reverseMapping()
    )

    # get line segment feats, needs cell1geometry vector
    line_dist_calculator = ncgp.Cell1LineSegmentDist2D()
    line_dist_feats_ = line_dist_calculator(
        cell_geometry[1]
    )

    # get geometry feats, needs cell1geometry, cell2geometr vectors
    # and cell1boundsvector
    geo_calculator = ncgp.Cell1BasicGeometricFeatures2D()
    geo_feats_ = geo_calculator(
        cell_geometry[1],
        cell_geometry[2],
        cell_bounds[1]
    )

    # get topology feats, needs bounds 0 and 1 and boundedBy 1 and 2
    # and cell1boundsvector
    topo_calculator = ncgp.Cell1BasicTopologicalFeatures2D()
    topo_feats_ = topo_calculator(
        cell_bounds[0],
        cell_bounds[1],
        cell_bounds[0].reverseMapping(),
        cell_bounds[1].reverseMapping()
    )

    # get the uv ids corresponding to the lines / faces
    # (== 1 cells), and map them back to the original ids
    line_uv_ids = np.array(cell_bounds[1])
    line_uv_ids = np.sort(
        replace_from_dict(line_uv_ids, reverse_mapping),
        axis=1
    )

    # assert (np.unique(line_uv_ids) == np.unique(uv_ids)).all()
    # find the mapping from the global uv-ids to the line-uv-ids
    matching_indices = find_matching_row_indices(uv_ids, line_uv_ids)
    edge_ids = matching_indices[:, 0]
    edge_ids_local = matching_indices[:, 1]

    n_edges = len(uv_ids)

    # take care of duplicates resulting from edges made up of multiple faces
    assert len(edge_ids) >= n_edges, "%i, %i" % (len(edge_ids), n_edges)
    unique_ids, unique_idx, face_counts = np.unique(
        edge_ids,
        return_index=True,
        return_counts=True
    )
    assert len(unique_ids) == n_edges

    # get the edges with multiple faces
    edges_w_multiple_faces = unique_ids[face_counts > 1]
    # get the face indices for each multi edge
    multi_indices = [edge_ids_local[edge_ids == multi_edge] for multi_edge in edges_w_multiple_faces]

    # get the edges with a single face
    edges_w_single_face_idx = edge_ids_local[unique_idx[face_counts == 1]]
    edges_w_single_face = unique_ids[face_counts == 1]

    # # for debugging
    # print "Edge-translators"
    # print edge_ids
    # print edge_ids_local
    # print "Single-edges"
    # print edges_w_single_face
    # print edges_w_single_face_idx
    # print "Multi-edges"
    # print edges_w_multiple_faces
    # print multi_indices

    # map the curvature features to the proper rag-edges
    # first for the edges with single face, then for edges with
    # multiple faces via average over the faces
    curve_feats = np.zeros((n_edges, curve_feats_.shape[1]), dtype='float32')
    curve_feats[edges_w_single_face] = curve_feats_[edges_w_single_face_idx]
    for jj, multi_edge in enumerate(edges_w_multiple_faces):
        curve_feats[multi_edge] = np.mean(curve_feats_[multi_indices[jj], :], axis=0)

    # map the line dist features to the proper rag-edges
    # first for the edges with single face, then for edges with
    # multiple faces via average over the faces
    line_dist_feats = np.zeros((n_edges, line_dist_feats_.shape[1]), dtype='float32')
    line_dist_feats[edges_w_single_face] = line_dist_feats_[edges_w_single_face_idx]
    for jj, multi_edge in enumerate(edges_w_multiple_faces):
        line_dist_feats[multi_edge] = np.mean(line_dist_feats_[multi_indices[jj], :], axis=0)

    # map the geometry features to the proper rag-edges
    # first for the edges with single face, then for edges with
    # multiple faces via average over the faces
    geo_feats = np.zeros((n_edges, geo_feats_.shape[1]), dtype='float32')
    geo_feats[edges_w_single_face] = geo_feats_[edges_w_single_face_idx]
    for jj, multi_edge in enumerate(edges_w_multiple_faces):
        geo_feats[multi_edge] = np.mean(geo_feats_[multi_indices[jj], :], axis=0)

    # map the topology features to the proper rag-edges
    # first for the edges with single face, then for edges with
    # multiple faces via average over the faces
    topo_feats = np.zeros((n_edges, topo_feats_.shape[1]), dtype='float32')
    topo_feats[edges_w_single_face] = topo_feats_[edges_w_single_face_idx]
    for jj, multi_edge in enumerate(edges_w_multiple_faces):
        topo_feats[multi_edge] = np.mean(topo_feats_[multi_indices[jj], :], axis=0)

    # face count features
    face_count_feats = np.ones(n_edges, dtype='float32')
    face_count_feats[edges_w_multiple_faces] = face_counts[edges_w_multiple_faces]

    feats = np.concatenate(
        [curve_feats, line_dist_feats, geo_feats, topo_feats, face_count_feats[:, None]],
        axis=1
    )

    return feats


# calculate topological features for xy-edges
# -> each edge corresponds to (potentially multiple)
# line faces and we calculate features via the mean
# over the line faces
# features: curvature, line distances, geometry, topology
def topo_feats_xy(rag, seg, edge_indications, node_z_coords):

    # number of features:
    # curvature features -> 33
    # line dist features -> 33
    # geometry features  -> 20
    # topology features  ->  6
    # faces per edge     ->  1
    n_feats = 93
    feats_xy = np.zeros((rag.numberOfEdges, n_feats), dtype='float32')
    uv_ids = rag.uvIds()

    # iterate over the slices and
    # TODO parallelize
    for z in xrange(seg.shape[0]):
        print "Slice", z

        # get the uv_ids in this slice
        nodes_z = np.where(node_z_coords == z)[0]
        uv_mask = find_exclusive_matching_indices(uv_ids, nodes_z)
        uv_ids_z = uv_ids[uv_mask]

        feats = topo_feats_slice(
            seg[z],
            uv_ids_z
        )
        feats_xy[uv_mask] = feats

    return feats_xy


# calculate topological features for z-edges
# -> each edge corresponds to an area that is bounded
# by line faces. we calculate features via statistics
# over the line faces
# features: curvature....
# TODO potential extra features: Union, IoU, segmentShape (= edge_area / edge_circumference)
def topo_feats_z(rag, seg, edge_indications):

    # number of features:
    # curvature features -> 33
    # line dist features -> 33
    # geometry features  -> 20
    # topology features  ->  6
    # faces per edge     ->  1
    n_feats = 93

    # get the uv-ids of z-edges
    assert rag.numberOfEdges == len(edge_indications), "%i, %i" % (rag.numberOfEdges, len(edge_indications))
    feats_z = np.zeros((rag.numberOfEdges, n_feats), dtype='float32')
    z_edge_ids = np.where(edge_indications == 0)[0]

    coordinate_calc = nrag.ragCoordinates(rag)
    assert coordinate_calc.storageLengths() == rag.numberOfEdges
    coordinates = {edge_id: coordinate_calc.edgeCoordinates(edge_id) for edge_id in z_edge_ids}
    start_coordinates = np.array([np.min(coord[:, 0]) for coord in coordinates.values()], dtype='uint32')
    assert len(start_coordinates) == len(z_edge_ids)

    # iterate over the pairs of adjacent slices, map z-edges to
    # a segmentation and compute the corresponding features
    for z in xrange(seg.shape[0] - 1):
        print "Slice", z

        # z-edges to segmentation:
        # first find the edges in connecting slice z to z + 1
        # then map them to a segmentation (note that z edges coresspond to areas !)
        this_edge_ids = z_edge_ids[start_coordinates == z]
        edge_seg = np.zeros(seg.shape[1:], dtype='uint32')

        # we map to a segmentation consecutive starting from 0
        for ii, edge_id in enumerate(this_edge_ids):
            coords = coordinates[edge_id][:, 1:]
            edge_seg[(coords[:, 0], coords[:, 1])] = ii

        # build a rag based on the edge segmentation to have the uv-ids
        # and extract features for the lines between the z-edges
        edge_rag = nrag.gridRag(edge_seg)
        edge_uv_ids = edge_rag.uvIds()
        feats_lines = topo_feats_slice(edge_seg, edge_uv_ids)

        # map the feats back to the z-edges via averaging over the line feats
        feats = np.zeros((len(this_edge_ids), feats_lines.shape[1]), dtype='float32')
        for ii in xrange(len(this_edge_ids)):
            edge_mask = (edge_uv_ids == ii).any(axis=1)
            feats[ii, :] = np.mean(feats_lines[edge_mask, :], axis=0)

        feats_z[this_edge_ids] = feats

    return feats_z


def topology_features_impl(rag, seg, edge_indications, edge_lens, node_z_coords):
    # calculate the topo features for xy and z edges
    # for now we use the same number if features here
    # if that should change, we need to pad with zeros
    print "Computing features for xy-edges..."
    feats_xy = topo_feats_xy(rag, seg, edge_indications, node_z_coords)
    print "Computing features for z-edges..."
    feats_z  = topo_feats_z(rag, seg, edge_indications)

    # merge features
    extra_features = np.zeros_like(feats_xy, dtype='float32')
    extra_features[edge_indications == 1] = feats_xy[edge_indications == 1]
    extra_features[edge_indications == 0] = feats_z[edge_indications == 0]

    extra_names = ['blub']  # TODO proper names
    return extra_features, extra_names


if __name__ == '__main__':
    # test the 'topology_features_xy_slice' function
    def test_topofeats_slice():
        seg = np.zeros((128, 128), dtype='uint32')
        seg[64:] = 1
        seg[32:96, 32:96] = 2
        rag = nrag.gridRag(seg)
        uv_ids = rag.uvIds()
        feats = topo_feats_slice(seg, uv_ids)
        print feats.shape
        print feats[:, -5:]

    def test_seg():
        import os
        home = os.path.expanduser('~')
        seg_ = vigra.readHDF5(
            os.path.join(home, 'seg_cgp1.h5'),
            'data'
        )
        seg_ -= 1
        seg = np.concatenate([
            seg_[None, :, :], seg_[None, :, :] + seg_.max() + 1],
            axis=0
        )
        assert seg.min() == 0
        rag = nrag.gridRag(seg)
        nodes_z = np.zeros(rag.numberOfNodes, dtype='uint32')
        nodes_z[seg[0]] = 0
        nodes_z[seg[1]] = 1
        uv_ids = rag.uvIds()
        edge_indications = (nodes_z[uv_ids[:, 0]] == nodes_z[uv_ids[:, 1]]).astype('uint8')
        assert len(edge_indications) == rag.numberOfEdges
        return seg, rag, nodes_z, edge_indications

    def test_topofeats_xy():
        seg, rag, nodes_z, edge_indications = test_seg()
        feats = topo_feats_xy(rag, seg, edge_indications, nodes_z)
        print feats.shape

    def test_topofeats_z():
        seg, rag, nodes_z, edge_indications = test_seg()
        feats = topo_feats_z(rag, seg, edge_indications)
        print feats.shape

    def topofeats():
        seg, rag, nodes_z, edge_indications = test_seg()
        edge_lens = np.ones(rag.numberOfEdges)
        feats, _ = topology_features_impl(rag, seg, edge_indications, edge_lens, nodes_z)
        print feats.shape

    # test_topofeats_xy()
    # test_topofeats_z()
    topofeats()
