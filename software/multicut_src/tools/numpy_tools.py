# wrapping some helpful numpy functionality
import numpy as np


# make the rows of array unique
# see http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
# TODO this could also be done in place
def get_unique_rows(array, return_index=False, return_inverse=False):
    array_view = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    if return_inverse:
        _, idx, inverse_idx = np.unique(array_view, return_index=True, return_inverse=True)
    else:
        _, idx = np.unique(array_view, return_index=True)
    unique_rows = array[idx]
    return_vals = (unique_rows,)
    if return_index:
        return_vals += (idx,)
    if return_inverse:
        return_vals += (inverse_idx,)
    return return_vals


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


# this returns a 1d array with the all the indices where a row of y matches a row in x
def find_matching_row_indices_fast(x, y):
    # In difference to find_matching_row_indices this finds just the positions in x but is significantly faster

    # This turns any array into 1D array
    x_bit = np.ascontiguousarray(x).view(
        np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    y_bit = np.ascontiguousarray(y).view(
        np.dtype((np.void, y.dtype.itemsize * y.shape[1])))

    return np.nonzero(np.in1d(x_bit, y_bit))[0]


# return the indices of the array which have at least one value from value list
def find_matching_indices(array, value_list):
    assert isinstance(array, np.ndarray)
    assert isinstance(value_list, np.ndarray)
    mask = np.in1d(array, value_list).reshape(array.shape)
    return np.where(mask.any(axis=1))[0]


# return the indices of the array which have only values from value list
def find_exclusive_matching_indices(array, value_list):
    assert isinstance(array, np.ndarray)
    assert isinstance(value_list, np.ndarray)
    mask = np.in1d(array, value_list).reshape(array.shape)
    return np.where(mask.all(axis=1))[0]


# numpy.replace: replcaces the values in array according to dict
# cf. SO: http://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
def replace_from_dict(array, dict_like):
    replace_keys, replace_vals = np.array(
        list(zip(*sorted(dict_like.items()))),
        dtype=array.dtype
    )
    indices = np.digitize(array, replace_keys, right=True)
    return replace_vals[indices]


# relabel a numpy segmentation conseutively
# -> reimplemantation of the vigra function that sometimes shows odd behaviour
# ignore value is hardcoded to 0 for now
def relabel_consecutive(array, start_label=0, keep_zeros=True):
    uniques = np.unique(array)
    new_values = np.arange(len(uniques), dtype=array.dtype)
    if start_label != 0:
        if keep_zeros:
            new_values[1:] += start_label
        else:
            new_values += start_label
    replace_dict = {uniques[i]: new_values[i] for i in xrange(len(new_values))}
    new_max = new_values.max()
    return replace_from_dict(array, replace_dict), new_max, replace_dict
