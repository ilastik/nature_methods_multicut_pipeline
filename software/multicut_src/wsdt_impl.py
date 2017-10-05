import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
import vigra
from concurrent import futures


# wrap vigra local maxima properly
def local_maxima(image, *args, **kwargs):
    assert image.ndim in (2, 3), "Unsupported dimensionality: {}".format(image.ndim)
    if image.ndim == 2:
        return vigra.analysis.localMaxima(image, *args, **kwargs)
    if image.ndim == 3:
        return vigra.analysis.localMaxima3D(image, *args, **kwargs)


# watershed on distance transform:
# seeds are generated on the inverted distance transform
# the probability map is used for growing
def compute_wsdt_segmentation(
    probability_map,
    threshold,
    sigma_seeds,
    min_segment_size=0,
    preserve_membrane=True
):

    # first, we compute the signed distance transform
    dt = signed_distance_transform(probability_map, threshold, preserve_membrane)

    # next, get the seeds via maxima on the (smoothed) distance transform
    seeds = seeds_from_distance_transform(dt, sigma_seeds)
    del dt  # remove the array name

    # run watershed on the pmaps with dt seeds
    segmentation = iterative_watershed(probability_map.astype('float32'), seeds, min_segment_size)
    return segmentation, segmentation.max()


def signed_distance_transform(probability_map, threshold, preserve_membrane):

    # get the distance transform of the pmap
    binary_membranes = (probability_map >= threshold)
    distance_to_membrane = distance_transform_edt(np.logical_not(binary_membranes))

    # Instead of computing a negative distance transform within the thresholded membrane areas,
    # Use the original probabilities (but inverted)
    if preserve_membrane:
        distance_to_membrane[binary_membranes] = -probability_map[binary_membranes]

    # Compute the negative distance transform and substract it from the distance transform
    else:
        distance_to_nonmembrane = distance_transform_edt(binary_membranes)

        # Combine the inner/outer distance transforms
        distance_to_nonmembrane[distance_to_nonmembrane > 0] -= 1
        distance_to_membrane[:] -= distance_to_nonmembrane

    return distance_to_membrane.astype('float32')


def seeds_from_distance_transform(distance_transform, sigma_seeds):

    # we are not using the dt after this point, so it's ok to smooth it
    # and later use it for calculating the seeds
    if sigma_seeds > 0.:
        distance_transform = vigra.filters.gaussianSmoothing(distance_transform, sigma_seeds)

    # If any seeds end up on the membranes, we'll remove them.
    # This is more likely to happen when the distance transform was generated with preserve_membrane_pmaps=True
    membrane_mask = (distance_transform < 0)

    seeds = local_maxima(distance_transform, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
    seeds = np.isnan(seeds).astype('uint32')
    seeds[membrane_mask] = 0

    return vigra.analysis.labelMultiArrayWithBackground(seeds)


def iterative_watershed(hmap, seeds, min_segment_size):

    seg, _ = vigra.analysis.watershedsNew(hmap, seeds=seeds)

    if min_segment_size:

        segments, counts = np.unique(seg, return_counts=True)

        # mask segments which are smaller than min_segment size
        mask = np.ma.masked_array(seg, np.in1d(seg, segments[counts < min_segment_size])).mask
        seg[mask] = 0

        seg, _ = vigra.analysis.watershedsNew(hmap, seeds=seg)

        # remove gaps in the list of label values.
        seg, _, _ = vigra.analysis.relabelConsecutive(seg, start_label=0, keep_zeros=False)

    return seg


def compute_stacked_wsdt(
    probability_map,
    threshold,
    sigma_seeds,
    min_segment_size=0,
    preserve_membrane=True
):
    seg = np.zeros_like(probability_map, dtype='uint32')

    def wsdt_z(z):
        ws, ws_max = compute_wsdt_segmentation(
            probability_map[z],
            threshold,
            sigma_seeds,
            min_segment_size=min_segment_size,
            preserve_membrane=preserve_membrane
        )
        seg[z] = ws
        return ws_max

    with futures.ThreadPoolExecutor(max_workers=8) as tp:
        tasks = [tp.submit(wsdt_z, z) for z in range(seg.shape[0])]
        offsets = [t.result() for t in tasks]

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets).astype(seg.dtype)

    seg += offsets[:, None, None]

    return seg


# FIXME this does not work yet, what we actually want is support for masked_arrays
def compute_wsdt_segmentation_with_mask(
    probability_map,
    mask,
    threshold,
    sigma_seeds,
    min_segment_size=0,
    preserve_membrane=True
):
    assert probability_map.shape == mask.shape
    seg = np.zeros_like(probability_map, dtype='uint32')
    seg_, seg_max = compute_wsdt_segmentation(
        probability_map[mask], threshold, sigma_seeds, min_segment_size, preserve_membrane
    )
    seg[mask] = (seg_ + 1)
    return seg, seg_max + 1


def compute_stacked_wsdt_with_mask(
    probability_map,
    mask,
    threshold,
    sigma_seeds,
    min_segment_size=0,
    preserve_membrane=True
):
    assert mask.shape == probability_map.shape
    seg = np.zeros_like(probability_map, dtype='uint32')

    def wsdt_z(z):
        ws, ws_max = compute_wsdt_segmentation_with_mask(
            probability_map[z],
            mask[z],
            threshold,
            sigma_seeds,
            min_segment_size=min_segment_size,
            preserve_membrane=preserve_membrane
        )
        seg[z] = ws
        return ws_max

    with futures.ThreadPoolExecutor(max_workers=8) as tp:
        tasks = [tp.submit(wsdt_z, z) for z in range(seg.shape[0])]
        offsets = [t.result() for t in tasks]

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets).astype(seg.dtype)

    for z in range(seg.shape[0]):
        seg[z][mask[z]] += offsets[z]

    return seg
