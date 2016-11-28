import numpy
import vigra
import networkx as nx

import logging
logger = logging.getLogger(__name__)

import time
import functools
def log_calls(log_level):
    def decorator( func ):
        @functools.wraps(func)
        def wrapper( *args, **kwargs ):
            try:
                start = time.time()
                logger.log(log_level, "{}...".format( func.__name__ ))
                return func(*args, **kwargs)
            finally:
                stop = time.time()
                logger.log(log_level, "{} took ({:0.2f} seconds)".format( func.__name__, stop - start ))
        wrapper.__wrapped__ = func # Emulate python 3 behavior of @functools.wraps
        return wrapper
    return decorator

@log_calls(logging.INFO)
def wsDtSegmentation(pmap, pmin, minMembraneSize, minSegmentSize, sigmaMinima, sigmaWeights, groupSeeds=True, growPmap=False, out_debug_image_dict=None, out=None):
    """A probability map 'pmap' is provided and thresholded using pmin.
    This results in a mask. Every connected component which has fewer pixel
    than 'minMembraneSize' is deleted from the mask. The mask is used to
    calculate the signed distance transformation.

    From this distance transformation the segmentation is computed using
    a seeded watershed algorithm. The seeds are placed on the local maxima
    of the distanceTransform after smoothing with 'sigmaMinima'.

    The weights of the watershed are defined by the inverse of the signed
    distance transform smoothed with 'sigmaWeights'.

    'minSegmentSize' determines how small the smallest segment in the final
    segmentation is allowed to be. If there are smaller ones the corresponding
    seeds are deleted and the watershed is done again.

    If 'groupSeeds' is True, multiple seed points that are clearly in the
    same neuron will be merged with a heuristik that ensures that no seeds of
    two different neurons are merged.

    If 'out_debug_image_dict' is not None, it must be a dict, and this function
    will save intermediate results to the dict as vigra.ChunkedArrayCompressed objects.

    Implementation Note: This algorithm has the potential to use a lot of RAM, so this
                         code goes attempts to operate *in-place* on large arrays whenever
                         possible, and we also delete intermediate results soon
                         as possible, sometimes in the middle of a function.
    """
    assert out_debug_image_dict is None or isinstance(out_debug_image_dict, dict)
    assert isinstance(pmap, numpy.ndarray), \
        "Make sure that pmap is numpy array, instead of: " + str(type(pmap))
    assert pmap.ndim in (2,3), "Input must be 2D or 3D.  shape={}".format( pmap.shape )

    distance_to_membrane = signed_distance_transform(pmap, pmin, minMembraneSize, out_debug_image_dict)
    binary_seeds = binary_seeds_from_distance_transform(distance_to_membrane, sigmaMinima, out_debug_image_dict)

    # seedsd from local minima
    #smoothed_for_local_minima = vigra.filters.gaussianSmoothing(pmap, 2.7)
    #localMinima = vigra.analysis.localMinima(smoothed_for_local_minima, alllowAtBorder = True, allowPlateaus = True, marker = numpy.nan)
    #seedsFromMinima = nimpy.isnan(localMinima)

    #binary_seeds += seedsFromMinima

    if groupSeeds:
        labeled_seeds = group_seeds_by_distance( binary_seeds, distance_to_membrane, out=out )
    else:
        labeled_seeds = vigra.analysis.labelMultiArrayWithBackground(binary_seeds.view(numpy.uint8), out=out)

    del binary_seeds
    save_debug_image('seeds', labeled_seeds, out_debug_image_dict)

    if sigmaWeights != 0.0:
        vigra.filters.gaussianSmoothing(distance_to_membrane, sigmaWeights, out=distance_to_membrane)
        save_debug_image('smoothed DT for watershed', distance_to_membrane, out_debug_image_dict)

    if growPmap:
        hmap = vigra.filters.gaussianSmoothing(pmap.astype('float32'),sigmaWeights)
        hmap += .1 * vigra.filters.gaussianSmoothing(pmap.astype('float32'),5*sigmaWeights)

        iterative_inplace_watershed(hmap, labeled_seeds, minSegmentSize, out_debug_image_dict)
    else:
        #thresholded_pmap = numpy.zeros_like(pmap, dtype = 'uint32')
        #thresholded_pmap[pmap < 0.5] = 1
        #print thresholded_pmap.shape
        #thresholded_ccs =vigra.analysis.labelMultiArrayWithBackground(thresholded_pmap.squeeze())
        #rag = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(pmap.shape), thresholded_ccs)
        #nodeFeatures = rag.accumulateNodeFeatures(pmap.astype('float32'))
        #nodeSeeds = numpy.zeros_like(nodeFeatures)
        #seedsMax = labeled_seeds.max() + 1
        #print "Num old seeds"
        #print seedsMax
        #for n in xrange(rag.nodeNum):
        #    if nodeFeatures[n] > 0.3 and nodeFeatures[n] < 0.7:
        #        nodeSeeds[n] = seedsMax
        #        seedsMax += 1
        #seedsNew = rag.projectLabelsToBaseGraph(nodeSeeds)

        #seeds = labeled_seeds + seedsNew
        #seeds = vigra.analysis.labelMultiArrayWithBackground(seeds.astype('uint32'))
        ##from volumina_viewer import volumina_n_layer
        ##volumina_n_layer([pmap,thresholded_pmap,thresholded_ccs,seedsNew,seeds])
        ##quit()

        #print "Num new seeds"
        #print seeds.max()
        ## Invert the DT: Watershed code requires seeds to be at minimums, not maximums
        distance_to_membrane[:] *= -1
        iterative_inplace_watershed(distance_to_membrane, labeled_seeds, minSegmentSize, out_debug_image_dict)
    return labeled_seeds


@log_calls(logging.INFO)
def wsDtSegmentationSpecial(pmap, pmin, minMembraneSize, minSegmentSize, sigmaMinima, sigmaWeights, ppitch, groupSeeds=True, out_debug_image_dict=None, out=None):
    assert out_debug_image_dict is None or isinstance(out_debug_image_dict, dict)
    assert isinstance(pmap, numpy.ndarray), \
        "Make sure that pmap is numpy array, instead of: " + str(type(pmap))
    assert pmap.ndim in (2,3), "Input must be 2D or 3D.  shape={}".format( pmap.shape )

    distance_to_membrane = signed_distance_transform(pmap, pmin, minMembraneSize, out_debug_image_dict, ppitch)
    offset = 0
    labeled_seeds = numpy.zeros_like(pmap, dtype = 'uint32')
    for z in xrange(distance_to_membrane.shape[2]):

        dist_z = distance_to_membrane[:,:,z]
        binary_seeds = binary_seeds_from_distance_transform(dist_z, sigmaMinima, out_debug_image_dict)

        if groupSeeds:
            labeled_seeds[:,:,z] = group_seeds_by_distance( binary_seeds, dist_z, out=out )
        else:
            labeled_seeds[:,:,z] = vigra.analysis.labelMultiArrayWithBackground(binary_seeds.view(numpy.uint8), out=out)

        del binary_seeds

        if sigmaWeights != 0.0:
            vigra.filters.gaussianSmoothing(dist_z, sigmaWeights, out=dist_z)

        # Invert the DT: Watershed code requires seeds to be at minimums, not maximums
        hmap = vigra.filters.gaussianSmoothing(pmap[:,:,z].astype('float32'),sigmaWeights)
        hmap += .1 * vigra.filters.gaussianSmoothing(pmap[:,:,z].astype('float32'),5*sigmaWeights)
        iterative_inplace_watershed(hmap, labeled_seeds[:,:,z], minSegmentSize, out_debug_image_dict)
        labeled_seeds[:,:,z] += offset
        offset = labeled_seeds.max()

    return labeled_seeds


@log_calls(logging.DEBUG)
def signed_distance_transform(pmap, pmin, minMembraneSize, out_debug_image_dict, ppitch = None):
    """
    Performs a threshold on the given image 'pmap' > pmin, and performs
    a distance transform to the threshold region border for all pixels outside the
    threshold boundaries (positive distances) and also all pixels *inside*
    the boundary (negative distances).

    The result is a signed float32 image.
    """
    # get the thresholded pmap
    binary_membranes = (pmap >= pmin).view(numpy.uint8)

    # delete small CCs
    labeled = vigra.analysis.labelMultiArrayWithBackground(binary_membranes)
    save_debug_image('thresholded membranes', labeled, out_debug_image_dict)
    del binary_membranes

    remove_wrongly_sized_connected_components(labeled, minMembraneSize, in_place=True)
    save_debug_image('filtered membranes', labeled, out_debug_image_dict)

    # perform signed dt on mask
    logger.debug("positive distance transform...")
    if ppitch != None:
        distance_to_membrane = vigra.filters.distanceTransform(labeled, pixel_pitch = ppitch)
    else:
        distance_to_membrane = vigra.filters.distanceTransform(labeled)

    # Save RAM with a sneaky trick:
    # Use distanceTransform in-place, despite the fact that the input and output don't have the same types!
    # (We can just cast labeled as a float32, since uint32 and float32 are the same size.)
    logger.debug("negative distance transform...")
    distance_to_nonmembrane = labeled.view(numpy.float32)
    if ppitch != None:
        vigra.filters.distanceTransform(labeled, background=False, out=distance_to_nonmembrane, pixel_pitch = ppitch)
    else:
        vigra.filters.distanceTransform(labeled, background=False, out=distance_to_nonmembrane, pixel_pitch = ppitch)
    del labeled # Delete this name, not the array

    # Combine the inner/outer distance transforms
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1
    distance_to_membrane[:] -= distance_to_nonmembrane

    save_debug_image('distance transform', distance_to_membrane, out_debug_image_dict)
    return distance_to_membrane

@log_calls(logging.DEBUG)
def binary_seeds_from_distance_transform(distance_to_membrane, smoothingSigma, out_debug_image_dict):
    """
    Return a binary image indicating the local maxima of the given distance transform.

    If smoothingSigma is provided, pre-smooth the distance transform before locating local maxima.
    """
    # Can't work in-place: Not allowed to modify input
    distance_to_membrane = distance_to_membrane.copy()

    if smoothingSigma != 0.0:
        distance_to_membrane = vigra.filters.gaussianSmoothing(distance_to_membrane, smoothingSigma, out=distance_to_membrane)
        save_debug_image('smoothed DT for seeds', distance_to_membrane, out_debug_image_dict)

    localMaximaND(distance_to_membrane, allowPlateaus=True, allowAtBorder=True, marker=numpy.nan, out=distance_to_membrane)
    seedsVolume = numpy.isnan(distance_to_membrane)

    save_debug_image('binary seeds', seedsVolume.view(numpy.uint8), out_debug_image_dict)
    return seedsVolume

@log_calls(logging.DEBUG)
def iterative_inplace_watershed(weights, seedsLabeled, minSegmentSize, out_debug_image_dict):
    """
    Perform a watershed over an image using the given seed image.
    The watershed is written IN-PLACE into the seed image.

    If minSegmentSize is provided, then watershed segments that were too small will be removed,
    and a second watershed will be performed so that the larger segments can claim the gaps.
    """
    _ws, max_label = vigra.analysis.watershedsNew(weights, seeds=seedsLabeled, out=seedsLabeled)

    if minSegmentSize:
        save_debug_image('initial watershed', seedsLabeled, out_debug_image_dict)
        remove_wrongly_sized_connected_components(seedsLabeled, minSegmentSize, in_place=True)
        _ws, max_label = vigra.analysis.watershedsNew(weights, seeds=seedsLabeled, out=seedsLabeled)

    logger.debug("Max Watershed Label: {}".format(max_label))

def vigra_bincount(labels):
    """
    A RAM-efficient implementation of numpy.bincount() when you're dealing with uint32 labels.
    If your data isn't int64, numpy.bincount() will copy it internally -- a huge RAM overhead.
    (This implementation may also need to make a copy, but it prefers uint32, not int64.)
    """
    labels = labels.astype(numpy.uint32, copy=False)
    labels = numpy.ravel(labels, order='K').reshape((-1, 1), order='A')
    # We don't care what the 'image' parameter is, but we have to give something
    image = labels.view(numpy.float32)
    counts = vigra.analysis.extractRegionFeatures(image, labels, ['Count'])['Count']
    return counts.astype(numpy.int64)

@log_calls(logging.DEBUG)
def remove_wrongly_sized_connected_components(a, min_size, max_size=None, in_place=False, bin_out=False):
    """
    Given a label image remove (set to zero) labels whose count is too low or too high.
    (Copied from lazyflow.)
    """
    original_dtype = a.dtype

    if not in_place:
        a = a.copy()
    if min_size == 0 and (max_size is None or max_size > numpy.prod(a.shape)): # shortcut for efficiency
        if (bin_out):
            numpy.place(a,a,1)
        return a

    component_sizes = vigra_bincount(a)
    bad_sizes = component_sizes < min_size
    if max_size is not None:
        numpy.logical_or( bad_sizes, component_sizes > max_size, out=bad_sizes )
    del component_sizes

    bad_locations = bad_sizes[a]
    a[bad_locations] = 0
    del bad_locations
    if (bin_out):
        # Replace non-zero values with 1
        numpy.place(a,a,1)
    return numpy.asarray(a, dtype=original_dtype)

@log_calls(logging.DEBUG)
def group_seeds_by_distance(binary_seeds, distance_to_membrane, out=None):
    """
    Label seeds in groups, such that every seed in each group is closer to at
    least one other seed in its group than it is to the nearest membrane.

    Parameters
    ----------
    binary_seeds
        A boolean image indicating where the seeds are

    distance_to_membrane
        A float32 image of distances to the membranes

    out
        Optional.  Must be uint32, same shape as binary_seeds.

    Returns
    -------
        A label image, uint32.
        Grouped seeds will have the same label value.
    """
    seed_locations = nonzero_coord_array(binary_seeds)
    assert seed_locations.shape[1] == binary_seeds.ndim
    num_seeds = seed_locations.shape[0]
    logger.debug("Number of seed points: {}".format(num_seeds))

    # Save RAM: shrink the dtype if possible
    if seed_locations.max() < numpy.sqrt(2**31):
        seed_locations = seed_locations.astype( numpy.int32 )

    # From the distance transform image, extract each seed's distance to the nearest membrane
    point_distances_to_membrane = distance_to_membrane[binary_seeds]

    # Create a graph of the seed points containing only the connections between 'close' seeds, as found below.
    # (Note that self->self edges are included in this graph, since that distance is 0.0)
    seed_graph = nx.Graph()

    # We'll find the distances between all points A and B,
    # but do it in batches since it takes a lot of RAM.
    # How big should the batches be?
    # We'll pick a batch size that requires about as much RAM as the original input data.
    # (RAM need per batch is 2*4*N*N bytes)
    orig_data_bytes = float(4*numpy.prod(binary_seeds.shape))
    batch_size = int(numpy.sqrt(orig_data_bytes / (2*4)))

    for batch_start_a in range( 0, num_seeds, batch_size ):
        batch_stop_a = min(batch_start_a + batch_size, num_seeds)
        point_batch_a = seed_locations[batch_start_a:batch_stop_a]
        distances_to_membrane_a = point_distances_to_membrane[batch_start_a:batch_stop_a]

        for batch_start_b in range( 0, num_seeds, batch_size ):
            batch_stop_b = min(batch_start_b + batch_size, num_seeds)
            point_batch_b = seed_locations[batch_start_b:batch_stop_b]
            distances_to_membrane_b = point_distances_to_membrane[batch_start_b:batch_stop_b]

            # Compute the distance of each seed in batch A to each seed in batch B
            pairwise_distances = pairwise_euclidean_distances(point_batch_a, point_batch_b)

            # Find the seed pairs that are closer to each other than either of them is to a membrane.
            close_pairs     = (pairwise_distances < distances_to_membrane_a[:, None])
            close_pairs[:] &= (pairwise_distances < distances_to_membrane_b[None, :])
            del pairwise_distances

            # Translate seed index within batch to index within entire seed list
            close_seed_indexes = nonzero_coord_array(close_pairs)
            close_seed_indexes[:,0] += batch_start_a
            close_seed_indexes[:,1] += batch_start_b

            # Update graph edges
            seed_graph.add_edges_from(close_seed_indexes)
            del close_seed_indexes

    del seed_locations
    del point_distances_to_membrane

    # Find the connected components in the graph, and give each CC a unique ID, starting at 1.
    logger.debug("Extracting connected seeds...")
    cc_start_time = time.time()
    seed_labels = numpy.zeros( (num_seeds,), dtype=numpy.uint32 )
    for group_label, grouped_seed_indexes in enumerate(nx.connected_components(seed_graph), start=1):
        for seed_index in grouped_seed_indexes:
            seed_labels[seed_index] = group_label
    del seed_graph
    logger.debug("... took {:2f} seconds".format( time.time() - cc_start_time ))

    # Apply the new labels to the original image
    labeled_seed_img = out
    if labeled_seed_img is None:
        labeled_seed_img = numpy.zeros( binary_seeds.shape, dtype=numpy.uint32 )
    else:
        labeled_seed_img[:] = 0
        assert labeled_seed_img.shape == binary_seeds.shape
        assert labeled_seed_img.dtype == numpy.uint32
    labeled_seed_img[binary_seeds] = seed_labels
    return labeled_seed_img

def pairwise_euclidean_distances( coord_array_a, coord_array_b ):
    """
    For all coordinates in the given arrays of shape (N, DIM) and (M, DIM),
    return an array of shape (N,M) of the distances
    of each item to all others.
    """
    N = len(coord_array_a)
    M = len(coord_array_b)
    assert coord_array_a.shape[-1] == coord_array_b.shape[-1]
    ndim = coord_array_a.shape[-1]

    distances = numpy.zeros((N,M), dtype=numpy.float32)
    for i in range(ndim):
        tmp = numpy.empty_like(distances) # force float32
        pairwise_subtracted = numpy.subtract.outer(coord_array_a[:,i], coord_array_b[:,i], out=tmp)
        squared = numpy.power(pairwise_subtracted, 2, out=pairwise_subtracted)
        distances[:] += squared

    numpy.sqrt(distances, out=distances)
    return distances

def nonzero_coord_array(a):
    """
    (Copied from lazyflow.utility.helpers)

    Equivalent to np.transpose(a.nonzero()), but much
    faster for large arrays, thanks to a little trick:
    The elements of the tuple returned by a.nonzero() share a common base,
    so we can avoid the copy that would normally be incurred when
    calling transpose() on the tuple.
    """
    base_array = a.nonzero()[0].base

    # This is necessary because VigraArrays have their own version
    # of nonzero(), which adds an extra base in the view chain.
    while base_array.base is not None:
        base_array = base_array.base
    return base_array

@log_calls(logging.DEBUG)
def localMaximaND(image, *args, **kwargs):
    """
    An ND wrapper for vigra's 2D/3D localMaxima functions.
    """
    assert image.ndim in (2,3), \
        "Unsupported dimensionality: {}".format( image.ndim )
    if image.ndim == 2:
        return vigra.analysis.localMaxima(image, *args, **kwargs)
    if image.ndim == 3:
        return vigra.analysis.localMaxima3D(image, *args, **kwargs)

@log_calls(logging.DEBUG)
def save_debug_image( name, image, out_debug_image_dict ):
    """
    If output_debug_image_dict isn't None, save the
    given image in the dict as a compressed array.
    """
    if out_debug_image_dict is None:
        return

    if hasattr(image, 'axistags'):
        axistags=image.axistags
    else:
        axistags = None

    out_debug_image_dict[name] = vigra.ChunkedArrayCompressed(image.shape, dtype=image.dtype, axistags=axistags)
    out_debug_image_dict[name][:] = image
