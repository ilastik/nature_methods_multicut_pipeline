
__author__ = "Nasim Rahaman"

__doc__ = """Utility functions to help with data logistics."""

import numpy as np
from scipy.io import loadmat
from scipy.ndimage.morphology import distance_transform_edt as edt
import h5py as h5
import itertools as it
import random
import cPickle as pkl

# Try to import dill if it's installed
try:
    import dill as dll
except ImportError:
    dll = None

import Antipasti.pykit as pyk

def ctrax2np(matpath):
    """
    Convert ctrax_results.mat to a numpy array of shape (T, numobjects, 2), where the last dimension comprises (x, y).
    :type matpath: str
    :param matpath: Path to mat file
    :rtype: numpy.ndarray
    """
    # Load matfile
    matfile = loadmat(matpath)
    # Compute the number of objects to be tracked
    uniqueobj, numoccurence = np.unique(matfile['identity'], return_counts=True)
    numobj = len(uniqueobj)
    # Check if the number of times an object occurs is same for all objects
    assert all([occnum == numoccurence[0] for occnum in numoccurence]), "Dissapearing objects found."
    # Compute the total number of frames
    numframes = numoccurence[0]
    # Preallocate
    parray = np.zeros(shape=(numframes, numobj, 2))
    # Loop over objects and assign
    for obj in range(numobj):
        parray[:, obj, 0], parray[:, obj, 1] = matfile['y_pos'][matfile['identity'] == obj], \
                                               matfile['x_pos'][matfile['identity'] == obj]
    return parray


def track2volume(track, fieldshape, ds=[1, 1], objectids=None, edtgain=1.):
    """
    Converts a track array (as returned by ctrax2np, say) to a volumetric block with EDT
    :type track: numpy.ndarray
    :param track: Track array

    :type fieldshape: list or tuple
    :param fieldshape: Shape of the tracking field

    :type ds: list or tuple
    :param ds: Downsampling ratio for lossless downsampling of track volume.

    :type objectids: list
    :param objectids: List of ID's of objects to be tracked

    :type edtgain: float
    :param edtgain: Gain of the negative exponential of euclidean distance transform

    :rtype: numpy.ndarray
    """

    assert ds[0] == ds[1], "Only square downsampling is supported for now."

    # Preallocate
    trackvol = np.ones(shape=(track.shape[0], fieldshape[0]/ds[0], fieldshape[1]/ds[1]))

    # Default for objectids
    objectids = range(trackvol.shape[1]) if objectids is None else objectids

    # Round track
    track = np.round(track/ds[0]).astype('int64')

    # Write to volume
    for obj in objectids:
        trackvol[range(track.shape[0]), track[range(track.shape[0]), obj, 0], track[range(track.shape[0]), obj, 1]] = 0.

    # This should have generated thread-like structures in the track volume. Run a negative exponential distance
    # transform on it and return
    return np.exp(np.array([-edtgain * ds[0] * edt(trackplane) for trackplane in trackvol]))


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


def toh5(data, path, datapath='data'):
    """
    Write `data` to a HDF5 volume.

    :type data: numpy.ndarray
    :param data: Data to write.

    :type path: str
    :param path: Path to the volume.

    :type datapath: str
    :param datapath: Path to the volume in the HDF5 volume.

    """
    with h5.File(path, 'w') as f:
        f.create_dataset(datapath, data=data)


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

    :type shuffle: bool
    :param shuffle: Whether to shuffle the iterator.
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


def pickle(obj, filename):
    # Write to file
    with open(filename, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def unpickle(filename):
    # Read from file
    with open(filename, 'r') as f:
        obj = pkl.load(f)
    # Return
    return obj


def dill(obj, filename):
    # Make sure dill is available
    assert dll is not None, "Dill could not be imported. Is it installed?"
    # Write to file
    with open(filename, 'wb') as f:
        dll.dump(obj, f, protocol=dll.HIGHEST_PROTOCOL)


def undill(filename):
    # Make sure dill is available
    assert dll is not None, "Dill could not be imported. Is it installed?"
    # Load from file
    with open(filename, 'r') as f:
        obj = dll.load(f)
    # Return
    return obj