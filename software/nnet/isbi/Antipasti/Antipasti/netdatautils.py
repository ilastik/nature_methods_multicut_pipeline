__author__ = "Nasim Rahaman"

__doc__ = """Utility functions to help with data logistics."""

import numpy as np
import h5py as h5
import itertools as it
import random

import pykit as pyk

# Function to load in a dataset from a h5file
def fromh5(path, datapath=None, dataslice=None, asnumpy=True, preptrain=None):
    """
    Opens a hdf5 file at path, loads in the dataset at datapath, and returns dataset as a numpy array.

    :type path: str
    :param path: Path to h5 file

    :type datapath: str
    :param datapath: Path in h5 file (of the dataset). If not provided, returns the first dataset found.

    :type asnumpy: bool
    :param asnumpy: Whether to return as a numpy array (or a h5 dataset object)

    :type preptrain: prepkit.preptrain
    :param preptrain: Train of preprocessing functions to be applied on the dataset before being returned
    """

    # Init file
    h5file = h5.File(path)
    # Init dataset
    h5dataset = h5file[datapath] if datapath is not None else h5file.values()[0]
    # Slice dataset
    h5dataset = h5dataset[dataslice] if dataslice is not None else h5dataset
    # Convert to numpy if required
    h5dataset = np.asarray(h5dataset) if asnumpy else h5dataset
    # Apply preptrain
    h5dataset = preptrain(h5dataset) if preptrain is not None else h5dataset
    # Close file
    h5file.close()
    # Return
    return h5dataset


# Define a sliding window iterator (this time, more readable than a wannabe one-liner)
def slidingwindowslices(shape, nhoodsize, stride=1, ds=1, window=None, ignoreborder=True, shuffle=True, rngseed=None,
                        startmins=None, startmaxs=None, shufflebuffersize=1000):
    """
    Returns a generator yielding (shuffled) sliding window slice objects.

    :type shape: int or list of int
    :param shape: Shape of the input data

    :type nhoodsize: int or list of int
    :param nhoodsize: Window size of the sliding window.

    :type stride: int or list of int
    :param stride: Stride of the sliding window.

    :type window: list
    :param window: Configure the sliding window. Examples:
                   With axistags 'yxz':
                       - window = ['x', 'x', 'x'] ==> 3D sliding windows over the 3D volume
                       - window = [[0, 1], 'x', 'x'] ==> 2D sliding window over the 0-th and 1-st xz planes
                       - window = ['x', [8, 9], 'x'] ==> 2D sliding window over the 8-th and 9-st yz planes

    :type ignoreborder: bool
    :param ignoreborder: Whether to skip border windows (i.e. windows without enough pixels to fill the specified
                         nhoodsize).

    :type shuffle: bool
    :param shuffle: Whether to shuffle the iterator.

    :type rngseed: int
    :param rngseed: Random number generator seed. Use to synchronize shuffled generators.

    :returns: A python generator whose next method yields a tuple of slices.
    """

    # Determine dimensionality of the data
    datadim = len(shape)

    # Parse window
    if window is None:
        window = ['x'] * datadim
    else:
        assert len(window) == datadim, "Window must have the same length as the number of data dimensions."

    # Parse nhoodsize and stride
    nhoodsize = [nhoodsize, ] * datadim if isinstance(nhoodsize, int) else nhoodsize
    stride = [stride, ] * datadim if isinstance(stride, int) else stride
    ds = [ds, ] * datadim if isinstance(ds, int) else ds

    # Seed RNG if a seed is provided
    if rngseed is not None:
        random.seed(rngseed)

    # Define a function that gets a 1D slice
    def _1Dwindow(startmin, startmax, nhoodsize, stride, ds, seqsize, shuffle):
        starts = range(startmin, startmax + 1, stride)

        if ignoreborder:
            slices = [slice(st, st + nhoodsize, ds) for st in starts if st + nhoodsize <= seqsize]
        else:
            slices = [slice(st, ((st + nhoodsize) if st + nhoodsize <= seqsize else None), ds) for st in starts]

        if shuffle:
            random.shuffle(slices)
        return slices

    # Get window start limits
    startmins = [0, ] * datadim if startmins is None else startmins
    startmaxs = [shp - nhoodsiz for shp, nhoodsiz in zip(shape, nhoodsize)] if startmaxs is None else startmaxs

    # The final iterator is going to be a cartesian product of the lists in nslices
    nslices = [_1Dwindow(startmin, startmax, nhoodsiz, st, dsample, datalen, shuffle) if windowspec == 'x'
               else [slice(ws, ws + 1) for ws in pyk.obj2list(windowspec)]
               for startmin, startmax, datalen, nhoodsiz, st, windowspec, dsample in zip(startmins, startmaxs, shape,
                                                                                nhoodsize, stride, window, ds)]

    return it.product(*nslices)


if __name__ == "__main__":

    # TEST PASSED
    # slicegen = slidingwindowslices([512, 512, 30], [256, 256, 1], stride=[256, 256, 1])

    # TEST PASSED
    # slicegen = slidingwindowslices([512, 512, 30], [256, 256, 1], stride=[256, 256, 1], window=['x', 'x', [13, 14]])

    slicegen = slidingwindowslices([512, 512, 30], [256, 256, 1], stride=[256, 256, 1], window=['x', 'x', [13, 14]])
    pass
