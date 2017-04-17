
#
# deprecated impl of curvature
# this never worked properly, because the coordinates must be
# sorted (which we can't just assume by default)
#

# features based on curvature of xy edges
@cacher_hdf5("feature_folder")
def curvature_features(self, seg_id):
    rag = self._rag(seg_id)
    curvature_feats = np.zeros( (rag.edgeNum, 4) )
    edge_ind = self.edge_indications(seg_id)
    for edge in xrange(rag.edgeNum):
        if edge_ind[edge] == 0:
            continue
        coords = rag.edgeCoordinates(edge)[:,:-1]
        try:
            dx_dt = np.gradient(coords[:,0])
        except IndexError as e:
            #print coords
            continue
        dy_dt = np.gradient(coords[:,1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        # curvature implemented after:
        # http://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt + dy_dt * dy_dt)**1.5

        curvature_feats[edge,0] = np.mean(curvature)
        curvature_feats[edge,1] = np.min(curvature)
        curvature_feats[edge,2] = np.max(curvature)
        curvature_feats[edge,3] = np.std(curvature)

    return np.nan_to_num(curvature_feats)
