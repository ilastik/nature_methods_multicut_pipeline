# wrapping some helpful numpy functionality
import numpy as np


# make the rows of array unique
# see http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
# TODO this could also be done in place
def get_unique_rows(array, return_index=False):
    array_view = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    _, idx = np.unique(array_view, return_index=True)
    unique_rows = array[idx]
    if return_index:
        return unique_rows, idx
    else:
        return unique_rows


# this returns a 2d array with the all the indices of matching rows for a and b
# cf. http://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
def find_matching_row_indices(x, y):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # using a dictionary, this is faster than the pure np variant
    indices = []
    rows_x = {tuple(row): i for i, row in enumerate(x)}
    for i, row in enumerate(y):
        if tuple(row) in rows_x:
            indices.append([rows_x[tuple(row)], i])
    return np.array(indices)

# return the indices of array which have at least one value from value list
def find_matching_indices(array, value_list):
    assert isinstance(array, np.ndarray)
    assert isinstance(value_list, np.ndarray)
    # reimplemented in cython for speed # TODO !!! include in conda package
    try:
        from cython_tools import find_matching_indices_fast
        return find_matching_indices_fast(array.astype('uint32'), value_list.astype('uint32'))
    except ImportError:
        #print "WARNING: Could not find cython function, using slow numpy version"
        # TODO this is the proper numpy way to do it, check if it is actually slower and get rid of cython functionality if this is fast enough
        # also, don't need to wrap this if it is just a one-liner
        mask = np.in1d(array, value_list).reshape(array.shape)
        return np.where(mask.all(axis=1))[0]

# numpy.replace: replcaces the values in array according to dict
# cf. SO: http://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
def replace_from_dict(array, dict_like):
    replace_keys, replace_vals = np.array(list(zip( *sorted(dict_like.items() ))))
    indices = np.digitize(array, replace_keys, right = True)
    return replace_vals[indices].astype(array.dtype)

