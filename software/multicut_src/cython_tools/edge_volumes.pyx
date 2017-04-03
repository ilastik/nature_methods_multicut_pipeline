import numpy as np
cimport cython
cimport numpy as np

from concurrent import futures

# typedefs
ctypedef np.uint32_t LabelType
PyLabelType = np.uint32

ctypedef np.int32_t CoordType
PyCoordType = np.int32

ctypedef np.uint32_t ValueType
PyValueType = np.uint32

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
    for z in xrange(shape[2]):
        for x in xrange(shape[0]):
            for y in xrange(shape[1]):
                l_u = seg[x,y,z]
                # check all nbrs in 4 nh
                for d in range(2):
                    coords_v[0],coords_v[1] = x, y
                    if coords_v[d] + 1 < shape[d]:
                        coords_v[d] += 1
                        l_v = seg[coords_v[0],coords_v[1],z]
                        if l_u != l_v:
                            try: # we may not have a corresponding edge-id due to defects
                                e_id = uv_id_dict[ (min(l_u,l_v), max(l_u,l_v)) ]
                                volume[x,y,z] = edge_labels[e_id]
                                volume[coords_v[0],coords_v[1],z] = edge_labels[e_id]
                            except KeyError:
                                continue
    return volume

# 
def fast_edge_volume_from_uvs_between_plane(
        np.ndarray[LabelType, ndim=3] seg,
        np.ndarray[LabelType, ndim=2] uv_ids,
        np.ndarray[ValueType, ndim=1] edge_labels,
        look_dn):

    cdef np.ndarray[ValueType, ndim=3] volume = np.zeros_like(seg, dtype = PyValueType)
    cdef int x, y, z, i, d
    cdef LabelType l_u, l_v

    # make a uv-> id to edge label dict
    uv_id_dict = { (uv[0],uv[1]) : i for i, uv in enumerate(uv_ids) }

    shape = volume.shape

    for x in xrange(shape[0]):
        for y in xrange(shape[1]):
            for z in xrange(shape[2]):
                if z + 1 < shape[2]:
                    l_u = seg[x,y,z]
                    l_v = seg[x,y,z+1]
                    try: # we may not have a corresponding edge-id due to defects and ignore mask
                        e_id = uv_id_dict[ (min(l_u,l_v), max(l_u,l_v)) ]
                        if look_dn:
                            volume[x,y,z+1] = edge_labels[e_id]
                        else:
                            volume[x,y,z] = edge_labels[e_id]
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
    
    for x in xrange(volume_dn.shape[0]):
        for y in xrange(volume_dn.shape[1]):
            l_u = seg_dn[x,y]
            l_v = seg_up[x,y]
            
            try: # we may not have a corresponding edge-id due to ignore mask
                e_id = uv_id_dict[ (min(l_u,l_v), max(l_u,l_v)) ]
                #print e_id
                #print edge_labels[e_id]
                volume_dn[x,y] = edge_labels[e_id]
                volume_up[x,y] = edge_labels[e_id]
            except KeyError:
                continue

    return volume_dn, volume_up
