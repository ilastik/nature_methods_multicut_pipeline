import numpy as np
cimport cython
cimport numpy as np
from copy import deepcopy

# typedefs
ctypedef np.uint32_t LabelType
PyLabelType = np.uint32

ctypedef np.int32_t CoordType
PyCoordType = np.int32

ctypedef np.uint8_t ValueType
PyValueType = np.uint8

# 
def fast_edge_volume_from_uvs_in_plane(
        np.ndarray[LabelType, ndim=3] seg,
        np.ndarray[LabelType, ndim=2] uv_ids,
        np.ndarray[ValueType, ndim=1] edge_labels):

    cdef np.ndarray[ValueType, ndim=3] volume = np.zeros_like(seg, dtype = PyValueType)
    cdef int x, y, z, i, d
    cdef LabelType l_u, l_v
    cdef np.ndarray[CoordType, ndim=1] coords_v = np.zeros(2, dtype = PyCoordType)

    # make a uv-> id to edge label dict
    uv_id_dict = { (uv[0],uv[1]) : i for i, uv in enumerate(uv_ids) }

    shape = volume.shape

    # dunno if this is a good idea in terms of cache locality (violating C-order!)
    # seems fast enough!
    for z in xrange(shape[0]):
        for y in xrange(shape[1]):
            for x in xrange(shape[2]):
                l_u = seg[z, y, x]
                # check all nbrs in 4 nh
                for d in range(1,3):
                    coords_v[0], coords_v[1] = y, x
                    if coords_v[d-1] + 1 < shape[d]:
                        coords_v[d-1] += 1
                        l_v = seg[z,coords_v[0],coords_v[1]]
                        if l_u != l_v:
                            try: # we may not have a corresponding edge-id due to defects
                                e_id = uv_id_dict[ (min(l_u,l_v), max(l_u,l_v)) ]
                                volume[z,y,x] = edge_labels[e_id]
                                volume[z,coords_v[0],coords_v[1]] = edge_labels[e_id]
                            except KeyError:
                                continue
    return volume


# 2 channles: 0 -> label from z to z+1
#             1 -> label from z+1 to z 
def fast_edge_volume_from_uvs_between_plane(
        np.ndarray[LabelType, ndim=3] seg,
        np.ndarray[LabelType, ndim=2] uv_ids,
        np.ndarray[ValueType, ndim=1] edge_labels
):

    shape = seg.shape

    # for some reason shape + (2,) does not compile in cython
    vol_shape = []
    for ii in range(3):
        vol_shape.append(shape[ii])
    vol_shape.append(2)
    vol_shape = tuple(vol_shape)

    cdef np.ndarray[ValueType, ndim=4] volume = np.zeros(vol_shape, dtype = PyValueType)
    cdef int x, y, z, d
    cdef LabelType l_u, l_v

    # make a uv-> id to edge label dict
    uv_id_dict = {(uv[0], uv[1]) : i for i, uv in enumerate(uv_ids)}

    for z in xrange(shape[0] - 1):
        for y in xrange(shape[1]):
            for x in xrange(shape[2]):
                l_u = seg[z, y, x]
                l_v = seg[z + 1, y, x]
                try: # we may not have a corresponding edge-id due to defects and ignore mask
                    e_id = uv_id_dict[ (min(l_u,l_v), max(l_u,l_v)) ]
                    volume[z + 1, y, x, 0] = edge_labels[e_id]
                    volume[z, y, x, 1] = edge_labels[e_id]
                except KeyError:
                    continue
    return volume


def fast_edge_volume_for_skip_edges_slice(
        np.ndarray[LabelType, ndim=2] seg_dn,
        np.ndarray[LabelType, ndim=2] seg_up,
        np.ndarray[LabelType, ndim=2] skip_uv_ids,
        np.ndarray[ValueType, ndim=1] edge_labels
        ):


    cdef np.ndarray[ValueType, ndim=2] volume_dn = np.zeros_like(seg_dn, dtype = PyValueType)
    cdef np.ndarray[ValueType, ndim=2] volume_up = np.zeros_like(seg_up, dtype = PyValueType)

    # make a uv-> id to edge label dict
    uv_id_dict = { (uv[0],uv[1]) : i for i, uv in enumerate(skip_uv_ids) }

    cdef LabelType l_u, l_v

    for y in xrange(volume_dn.shape[1]):
        for x in xrange(volume_dn.shape[2]):
            l_u = seg_dn[y,x]
            l_v = seg_up[y,x]

            try: # we may not have a corresponding edge-id due to ignore mask
                e_id = uv_id_dict[ (min(l_u,l_v), max(l_u,l_v)) ]
                #print e_id
                #print edge_labels[e_id]
                volume_dn[y,x] = edge_labels[e_id]
                volume_up[y,x] = edge_labels[e_id]
            except KeyError:
                continue

    return volume_dn, volume_up
