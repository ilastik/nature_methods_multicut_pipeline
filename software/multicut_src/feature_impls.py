import numpy as np
import vigra

from tools import find_matching_row_indices, replace_from_dict, find_exclusive_matching_indices

# from concurrent import futures
# from ExperimentSettings import ExperimentSettings

# if build from source and not a conda pkg, we assume that we have cplex
try:
    import nifty
    import nifty.cgp as ncgp
except ImportError:
    try:
        import nifty_with_cplex as nifty  # conda version build with cplex
        import nifty_with_cplex.cgp as ncgp
    except ImportError:
        try:
            import nifty_with_gurobi as nifty  # conda version build with gurobi
            import nifty_with_gurobi.cgp as ncgp
        except ImportError:
            raise ImportError("No valid nifty version was found.")


# calculate topological features for xy-edges
# -> each edge corresponds to (potentially multiple)
# line faces and we calculate features via the mean
# over the line faces
# features: curvature, line distances, geometry, topology
def topo_feats_xy(rag, seg, node_z_coords):

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

        # get the uv_ids in this slice
        nodes_z = np.where(node_z_coords == z)[0]
        uv_mask = find_exclusive_matching_indices(uv_ids, nodes_z)
        uv_ids_z = uv_ids[uv_mask]

        # get the segmentation in this slice and map it
        # to a consecutive labeling starting from 1
        seg_z, _, mapping = vigra.analysis.relabelConsecutive(seg[z], start_label=1, keep_zeros=False)
        reverse_mapping = {new: old for old, new in mapping.iteritems()}
        assert seg_z.min() == 1

        # print seg_z.shape
        # vigra.writeHDF5(seg_z, '/home/constantin/seg_cgp.h5', 'data', compression='gzip')

        tgrid = ncgp.TopologicalGrid2D(seg_z)

        # extract the cell geometry
        cell_geometry = tgrid.extractCellsGeometry()
        cell_bounds = tgrid.extractCellsBounds()

        # get curvature feats, needs cell1geometry vector and cell1boundedby vector
        curvature_calculator = ncgp.Cell1CurvatureFeatures2D()
        curve_feats = curvature_calculator(
            cell_geometry[1],
            cell_bounds[0].reverseMapping()
        )

        # get line segment feats, needs cell1geometry vector
        line_dist_calculator = ncgp.Cell1LineSegmentDist2D()
        line_dist_feats = line_dist_calculator(
            cell_geometry[1]
        )

        # get geometry feats, needs cell1geometry, cell2geometr vectors
        # and cell1boundsvector
        geo_calculator = ncgp.Cell1BasicGeometricFeatures2D()
        geo_feats = geo_calculator(
            cell_geometry[1],
            cell_geometry[2],
            cell_bounds[1]
        )

        # get topology feats, needs bounds 0 and 1 and boundedBy 1 and 2
        # and cell1boundsvector
        topo_calculator = ncgp.Cell1BasicTopologicalFeatures2D()
        topo_feats = topo_calculator(
            cell_bounds[0],
            cell_bounds[1],
            cell_bounds[0].reverseMapping(),
            cell_bounds[1].reverseMapping()
        )

        # get the uv ids corresponding to the lines / faces
        # (== 1 cells), and map them back to the original ids
        line_uv_ids = np.array(cell_bounds[1])
        line_uv_ids = replace_from_dict(line_uv_ids, reverse_mapping)

        # assert (np.unique(line_uv_ids) == np.unique(uv_ids_z)).all()
        edge_ids = find_matching_row_indices(uv_ids_z, line_uv_ids)[:, 0]

        # take care of duplicates resulting from edges made up of multiple faces
        assert len(edge_ids) >= len(uv_ids_z), "%i, %i" % (len(edge_ids), len(uv_ids_z))

        unique_ids, face_counts = np.unique(edge_ids, return_counts=True)
        print unique_ids[face_counts > 1]
        print face_counts[face_counts > 1]

        quit()

        # TODO face_counts
        feats = np.concatenate(
            [curve_feats, line_dist_feats, geo_feats, topo_feats],
            axis=1
        )
        feats = feats[edge_ids]

        feats_xy[edge_ids] = feats

    return feats_xy


# calculate topological features for z-edges
# -> each edge corresponds to an area that is bounded
# by line faces. we calculate features via statistics
# over the line faces
# features: curvature....
# TODO potential extra features: Union, IoU, segmentShape (= edge_area / edge_circumference)
def topo_feats_z(rag, seg, edge_indications):
    pass
    # iterate over the pairs of adjacent slices, map z-edges to
    # a segmentation and compute the corresponding features


def topology_features_impl(rag, seg, edge_indications, edge_lens, node_z_coords):
    # calculate the topo features for xy and z edges
    # for now we use the same number if features here
    # if that should change, we need to pad with zeros
    feats_xy = topo_feats_xy(rag, seg, node_z_coords)
    feats_z  = topo_feats_z(rag, seg, edge_indications)

    # merge features
    extra_features = np.zeros_like(feats_xy, dtype='float32')
    extra_features[edge_indications == 1] = feats_xy[edge_indications == 1]
    extra_features[edge_indications == 0] = feats_z[edge_indications == 0]

    extra_names = ['blub']  # TODO proper names
    return extra_features, extra_names


# if __name__ == '__main__':
#     import nifty.graph.rag as nrag
#     seg_z = vigra.readHDF5('/home/consti/seg_cgp.h5', 'data') - 1
#     seg_z2 = seg_z + seg_z.max() + 1
#     seg = np.concatenate(
#         [seg_z[None, :], seg_z2[None, :]],
#         axis=0
#     )
#     print seg.shape
#     rag = nrag.gridRag(seg)
#     edge_lens = np.ones(rag.numberOfEdges, dtype='uint32')
#     nodes_z = np.zeros(rag.numberOfNodes)
#     nodes_z[seg[0]] = 0
#     nodes_z[seg[1]] = 1
#     uv_ids = rag.uvIds()
#     edge_indications = nodes_z[uv_ids[:, 0]] == nodes_z[uv_ids[:, 1]].astype('uint8')
#     topology_features_impl(rag, seg, edge_indications, edge_lens)
