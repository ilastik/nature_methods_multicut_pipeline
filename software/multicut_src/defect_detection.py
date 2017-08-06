from __future__ import print_function, division

import numpy as np
from concurrent import futures

#
# Defect detection
#
# -> this is not coupled to the dataset any longer


def oversegmentation_statistics(seg, n_bins):
    """
    Get the superpixel statistics over slices.
    """

    def extract_segs_in_slice(z):
        # 2d blocking representing the patches
        seg_z = seg[:, :, z]
        return np.unique(seg_z).shape[0]

    # parallel
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = []
        for z in range(seg.shape[2]):
            tasks.append(executor.submit(extract_segs_in_slice, z))
        segs_per_slice = [fut.result() for fut in tasks]

    # calculate histogram to have a closer look at the stats
    histo, bin_edges = np.histogram(segs_per_slice, bins=n_bins)
    # we only need the bin_edges
    return bin_edges


def defect_slice_detection(seg, n_bins, bin_threshold):
    """
    Find defected slices based on the superpixel statistics.
    Returns a mask showing the defects.
    """

    bin_edges = oversegmentation_statistics(seg, n_bins)

    threshold = bin_edges[bin_threshold]
    out = np.zeros_like(seg, dtype='uint8')

    def detect_defected_slice(z):
        seg_z = seg[:, :, z]
        # get number of segments for patches in this slice
        n_segs = np.unique(seg_z).shape[0]
        # threshold for a defected slice
        if n_segs < threshold:
            out[:, :, z] = 1
            return True
        else:
            return False

    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = []
        for z in range(seg.shape[2]):
            tasks.append(executor.submit(detect_defected_slice, z))
        defect_indications = [fut.result() for fut in tasks]

    # report the defects
    for z in range(seg.shape[2]):
        if defect_indications[z]:
            print("DefectSliceDetection: slice %i is defected." % z)

    return out
