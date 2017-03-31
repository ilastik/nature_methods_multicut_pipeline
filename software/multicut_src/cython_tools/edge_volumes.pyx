import numpy as np
cimport cython
cimport numpy as np
from libcpp cimport bool

# typedefs
ctypedef np.uint32_t LabelType
PyLabelType = np.uint32

ctypedef np.int32_t CoordType
PyCoordType = np.int32

ctypedef np.uint32_t ValueType
PyValueType = np.uint32

# 
def fast_edge_volume_from_uvs(
        np.ndarray[LabelType, ndim=3] seg,
        np.ndarray[LabelType, ndim=2] uv_ids,
        np.ndarray[ValueType, ndim=1] edge_labels,
        bool ignore_zeros = True):

    cdef np.ndarray[ValueType, ndim=3] volume = np.zeros_like(seg, dtype = PyValueType)
    cdef int x, y, z, i, d
    cdef np.ndarray[CoordType, ndim=1] coords_u = np.zeros(3, dtype = PyCoordType)
    cdef np.ndarray[CoordType, ndim=1] coords_v = np.zeros(3, dtype = PyCoordType)

    # make a uv-> id to edge label dict
    uv_id_dict = {(u,v) : i for i, u, v in enumerate(uv_ids) }

    shape = volume.shape
    for x in xrange(shape[0]):
        for y in xrange(shape[1]):
            for z in xrange(shape[2]):

                l_u = seg[x,y,z]
                # this should be possibe in a less ug
                coords_u[0],coords_u[1],coords_u[2] = x, y, z

                # check all nbrs in 6 nh
                for d in range(3):
                    coords_v[0],coords_v[1],coords_v[2] = x, y, z
                    if coords_u[d] + 1 < shape[d]:
                        coords_v[d] += 1
                        l_v = seg[coords_v]
                    if l_u != l_v:
                        e_id = uv_id_dict[ (min(l_u,l_v), max(l_u,l_v)) ]
                        volume[coords_u] = edge_labels[e_id]
                        volume[coords_v] = edge_labels[e_id]
    return volume
