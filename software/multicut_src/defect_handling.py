import vigra
import numpy as np
from concurrent import futures

from DataSet import DataSet

@cacher_hdf5
def oversegmentation_statistics(ds, seg_id, n_bins):
    #seg = ds.seg(seg_id)
    pass


@cacher_hdf5
def defect_slice_detection(ds, seg_id, bin_threshold):
    pass


@cacher_hdf5
def defects_to_nods(ds, seg_id, n_bins, bin_threshold):
    pass


@cacher_hdf5
def modified_adjacency(ds, seg_id, n_bins, bin_threshold):
    pass


# TODO modified features, need to figure out how to do this exactly ...
def _get_replace_slices(slice_list):
    # find consecutive slices with defects
    consecutive_defects = np.split(slice_list, np.where(np.diff(defected_slices) != 1)[0] + 1)
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


@cacher_hdf5
def postprocess_segmentation(seg_result, slice_list):
    pass


@cacher_hdf5
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
