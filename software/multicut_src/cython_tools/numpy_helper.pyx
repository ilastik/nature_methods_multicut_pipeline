import numpy as np
cimport cython
from cython.view cimport array as cvarray
cimport numpy as np

# typedefs
ctypedef np.uint32_t DataType
PyDataType = np.uint32

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline list _find_indices(
        DataType[:,:] array,
        DataType[:] compare
        ):
    
    cdef list indices = []
    cdef size_t xx, yy, cc
    cdef int match_found# bool workaround... FIXME
    n_row = array.shape[0]
    n_col = array.shape[1]
    n_compare = compare.shape[0]
    for xx in range(n_row):
        match_found = 0
        for yy in range(n_col):
            if match_found == 1:
                indices.append(xx)
                break
            for cc in range(n_compare):
                if array[xx,yy] == compare[cc]:
                    match_found = 1
                    break
    return indices

def find_matching_indices_fast(
        np.ndarray[DataType, ndim=2] x,
        np.ndarray[DataType, ndim=1] y):
    return np.array( _find_indices(x, y), dtype = PyDataType)
