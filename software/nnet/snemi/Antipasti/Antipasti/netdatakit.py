import glob
import re

__author__ = 'nasimrahaman'

""" Module to handle data handling. """

import numpy as np
import scipy.ndimage as ndi
import h5py as h5
from sklearn.preprocessing import scale
import prepkit as pk
import netdatautils as ndl
import itertools as it
import random
import copy
import pickle as pkl
import os
import pykit as pyk


# Class to process volumetric data from HDF5
class cargo:
    def __init__(self, h5path=None, pathh5=None, data=None, axistags=None, batchsize=20, nhoodsize=None, ds=None,
                 window=None, stride=None, preload=False, dataslice=None, preptrain=None, shuffleiterator=True):
        """
        :type h5path: str
        :param h5path: Path to h5 file

        :type pathh5: str
        :param pathh5: Path to dataset in h5 file

        :type data: numpy.ndarray
        :param data: Volumetric data (if no HDF5 given)

        :type axistags: str or list
        :param axistags: Specify which dimensions belong to what. For data 3D, axistag='cij' would imply that the first
                         dimension is channel, second and third i and j respectively.
                         Another example: axistag = 'icjk' --> dim0: i, dim1: channel, dim2:j, dim3: k
                         Note: 2D datasets are batched as (batchsize, channel, i = y, j = x)
                               3D datasets are batched as (batchsize, k = z, channel, i = y, j = x)

        :type batchsize: int
        :param batchsize: Batch size

        :type nhoodsize: list
        :param nhoodsize: Size of the sliding window. Use None for a channel dimensions (see doc axistags).

        :type ds: list
        :param ds: Downsampling ratio. Note that downsampling can also be done as preprocessing, but that would
                   require the upsampled data to be loaded to RAM beforehand.

        :type window: list
        :param window: Specify the sliding window to use while iterating over the dataset.
                       Example: for axistag = ijk:
                                ['x', 'x', [1, 4, 8]] makes a sliding window over the ij planes k = 1, 4 and 8.
                                ['x', [3, 5, 1], 'x'] makes a sliding window over the ik planes j = 3, 5 and 1.
                       Example: for axistag = cijk:
                                ['x', 'x', 'x', [1, 2]] makes a sliding window over ij planes k = 1, and 2.
                                [[1, 2], 'x', 'x', 'x'] makes no sliding window (i.e. channel axis ignored).
                       NOTE: Don't worry about 'x' corresponding to a channel dimension.

        :type stride: list
        :param stride: Stride of the sliding window

        :type preload: bool
        :param preload: Whether to preload data to RAM. Proceed with caution for large datasets!

        :type dataslice: tuple of slice
        :param dataslice: How to slice HDF5 data before building the generator. Must be a tuple of slices.
                          Ignores axistags.
                          Example: dataslice = (slice(0, 700), slice(0, 600), slice(0, 500)) would replace the data
                          loaded from the HDF5 file (or the one provided) with data[0:700, 0:600, 0:500].

        :type preptrain: pk.preptrain
        :param preptrain: Train of preprocessing functions

        :type shuffleiterator: bool
        :param shuffleiterator: Whether to shuffle batchstream generator
        """

        # Meta
        self.h5path = h5path
        self.pathh5 = pathh5
        self.batchsize = batchsize
        self.preptrain = preptrain
        self.shuffleiterator = shuffleiterator
        self.rngseed = random.randint(0, 100)

        # Read h5 file and dataset, fetch input dimension if h5path and pathh5 given
        if h5path is not None and h5path is not None:
            h5file = h5.File(h5path)
            if not preload:
                self.data = h5file[pathh5]
            else:
                self.data = h5file[pathh5][:]

        elif data is not None:
            self.data = data
        else:
            raise IOError('Path to or in h5 data not found and no data variable given.')

        # Slice data if required
        if dataslice is not None:
            self.data = self.data[tuple(dataslice)]

        # Parse window
        self.window = (['x'] * len(self.data.shape) if not window else window)
        assert len(self.window) == len(self.data.shape), \
            "Variable window must be a list with len(window) = # data dimensions"

        # Determine data dimensionality
        self.datadim = len(self.data.shape)

        # Parse axis tags
        if axistags is not None:
            assert len(axistags) == self.datadim, "Axis tags must be given for all dimensions."
            assert all(axistag in ['i', 'j', 'k', 'c'] for axistag in axistags) and len(axistags) == len(set(axistags)), \
                "Axis tags may only contain the characters c, i, j and k without duplicates."
            self.axistags = list(axistags)
        elif self.datadim == 2:
            self.axistags = list('ij')
        elif self.datadim == 3:
            self.axistags = list('ijk')
        elif self.datadim == 4:
            self.axistags = list('cijk')
        else:
            raise NotImplementedError("Supported data dimensions: 2, 3 and 4")

        # Parse downsampling ratio
        # Set defaults
        self.ds = ([(4 if windowspec == 'x' else 1) for windowspec in self.window]
                   if not ds else [nds if self.window[n] == 'x' else 1
                                   for n, nds in enumerate(ds)])

        # Check for length/dimensionality inconsistencies
        assert len(self.ds) == len(self.window)

        # Parse nhoodsize
        # Set defaults:
        self.nhoodsize = ([(100 if windowspec == 'x' else 1) for windowspec in self.window]
                          if not nhoodsize else [nnhs if self.window[n] == 'x' else 1
                                                 for n, nnhs in enumerate(nhoodsize)])

        # Check for length/dimensionality inconsistencies
        assert len(self.nhoodsize) == len(self.window)

        # Replace nhoodsize at channel axis with axis length (this is required for the loop to work)
        if 'c' in self.axistags:
            self.nhoodsize[self.axistags.index('c')] = self.data.shape[self.axistags.index('c')]

        # Determine net dimensions from axistag
        self.netdim = len(self.axistags) - self.axistags.count('c') - len(self.window) + self.window.count('x')

        # Parse strides
        self.stride = ([1] * len(self.data.shape) if not stride else stride)

        # Generate batch index generator
        self.batchindexiterator = None
        self.restartgenerator()

    # Generator to stream batches from HDF5 (after applying preprocessing pipeline)
    def batchstream(self):

        while True:
            # Fetch batch slices
            batchslices = []
            break_ = False
            for _ in range(self.batchsize):
                try:
                    batchslices.append(self.batchindexiterator.next())
                except StopIteration:
                    break_ = True

            if len(batchslices) == 0 and break_:
                raise StopIteration

            # Fetch slices from data
            rawbatch = np.array([self.data[tuple(batchslice)] for batchslice in batchslices])

            # Transform batch to the correct shape using the provided axistags
            rawbatch = self.transformbatch(rawbatch)

            # Apply preptrain
            if self.preptrain is not None:
                ppbatch = self.preptrain(rawbatch)
            else:
                ppbatch = rawbatch

            # Stream away
            yield ppbatch

            if break_:
                raise StopIteration

    # Method to transform fetched batch to a format NNet accepts
    def transformbatch(self, batch):

        # Insert a singleton dimension as placeholder for c if no channel axis specified and accordingly reformat. This
        # singleton dimension may or may not be squeezed out later, but that is to be taken care of.
        # axis tags
        if 'c' not in self.axistags:
            # Switch to indicate if 'c' in axistags
            channelnotinaxistags = True
            # Add singleton dimension
            batch = batch[:, None, ...]
            # Format axis tag accordingly
            axistags = (['b'] if 'b' not in self.axistags else []) + ['c'] + self.axistags
        else:
            channelnotinaxistags = False
            # Add b to axis tag if not present already
            axistags = (['b'] if 'b' not in self.axistags else []) + self.axistags

        # Transform 2D
        if self.datadim == 2:
            batch = np.transpose(batch, axes=[axistags.index('b'), axistags.index('c'),
                                              axistags.index('i'), axistags.index('j')])
            # Squeeze batch. This may or may not take out the 'c' dimension, depending on whether it's a singleton
            # dimension; in case it does, we'll add it back in
            # NOTE: Batch dimension is squeezed out if batchsize = 1. Squeeze samplewise.
            batch = np.array([sample.squeeze() for sample in batch])
            if channelnotinaxistags:
                batch = batch[:, None, ...]

        # Transform 3D
        elif self.datadim == 3:
            batch = np.transpose(batch, axes=[axistags.index('b'), axistags.index('k'),
                                              axistags.index('c'), axistags.index('i'), axistags.index('j')])
            # Squeeze batch. This may or may not take out the 'c' dimension, depending on whether it's a singleton
            # dimension; in case it does, we'll add it back in
            # NOTE: Batch dimension is squeezed out if batchsize = 1. Squeeze samplewise.
            batch = np.array([sample.squeeze() for sample in batch])
            if channelnotinaxistags and self.netdim == 3:
                batch = batch[:, :, None, ...]
            elif channelnotinaxistags and self.netdim == 2:
                batch = batch[:, None, ...]

        return batch

    # Method to restart main (iteration) generators
    def restartgenerator(self, rngseed=None):

        # Parse rngseed
        if not rngseed:
            rngseed = self.rngseed

        self.batchindexiterator = ndl.slidingwindowslices(self.data.shape, self.nhoodsize, self.stride, self.ds,
                                                          self.window, shuffle=self.shuffleiterator, rngseed=rngseed)

    # Method to clone cargo crate (i.e. cargo with a different dataset)
    def clonecrate(self, h5path=None, pathh5=None, data=None, syncgenerators=False):

        # Parse
        h5path = (self.h5path if not h5path else h5path)
        pathh5 = (self.pathh5 if not pathh5 else pathh5)
        data = (self.data if data is None else data)

        # Check if the file shapes consistent (save for another day)
        pass

        # FIXME: Sync shuffled generators
        # Make new cargo object
        newcargo = cargo(h5path=h5path, pathh5=pathh5, data=data, axistags=self.axistags, batchsize=self.batchsize,
                         nhoodsize=self.nhoodsize, ds=self.ds, window=self.window, stride=self.stride,
                         shuffleiterator=self.shuffleiterator)

        # Sync generators if requested
        if syncgenerators:
            self.syncgenerators(newcargo)

        # return
        return newcargo

    # Method to copy a cargo object
    def copy(self, syncgenerators=True):
        # Make new cargo
        newcargo = cargo(h5path=self.h5path, pathh5=self.pathh5, data=self.data, axistags=self.axistags,
                         batchsize=self.batchsize, nhoodsize=self.nhoodsize, ds=self.ds, window=self.window,
                         stride=self.stride,
                         preptrain=self.preptrain, shuffleiterator=True)

        # sync generators if requested
        if syncgenerators:
            self.syncgenerators(newcargo)

        # and return
        return newcargo

    # Method to synchronize the generators of self and a given cargo instance
    def syncgenerators(self, other):
        """
        :type other: cargo
        """
        # Set rngseeds
        other.rngseed = self.rngseed
        # Restart both generators
        self.restartgenerator()
        other.restartgenerator()

    # Define self as an iterator
    def __iter__(self):
        return self

    # Define next method
    def next(self):
        # FIXME The fuck were you thinking?
        return self.batchstream().next()



class masker:
    """ Class to mask a cargo with the other """

    def __init__(self, datacargo, maskcargo, maskfill='noise', maskthreshold=1, reversemaskpolarity=False):
        """
        :type datacargo: cargo
        :param datacargo: Dataset to be masked (wrapped in cargo)

        :type maskcargo: cargo
        :param maskcargo: Mask dataset (wrapped in cargo)

        :type maskfill: str
        :param maskfill: What to fill the mask negatives with. Options: 'noise' (Gaussian noise) or 'zeros'.

        :type maskthreshold: float
        :param maskthreshold: Threshold to apply on mask. 1 => No threshold applied.

        :type reversemaskpolarity: bool
        :param reversemaskpolarity: Whether to reverse mask polarity (i.e. set mask = 1 - mask)
        """

        # Assertions
        assert isinstance(datacargo, cargo), "Datacargo must be an instance of netdatakit.cargo."
        assert isinstance(maskcargo, cargo), "Maskcargo must be an instance of netdatakit.cargo, " \
                                             "preferably a clone (via netdatakit.cargo.clonecrate)."
        assert maskthreshold > 0, "Mask threshold must be larger than 0!"

        # Meta
        self.datacargo = datacargo
        self.maskcargo = maskcargo
        self.maskfill = maskfill
        self.reversemaskpolarity = reversemaskpolarity
        self.maskthreshold = maskthreshold

        # Generate generator
        self.batchindexiterator = zip(self.datacargo.batchstream(), self.maskcargo.batchstream())

    # Method to do the actual masking
    def maskbatch(self, databatch, maskbatch):

        # Soft mask
        if self.maskthreshold == 1:
            # Mask and fill
            if self.maskfill == 'noise':
                return maskbatch * databatch + (1 - maskbatch) * np.random.uniform(low=0., high=1.,
                                                                                   size=databatch.shape)
            elif self.maskfill == 'zeros':
                return maskbatch * databatch
            else:
                raise NotImplementedError("Mask fill not implemented.")
        # Hard mask
        else:
            # Threshold
            maskbatch[maskbatch < self.maskthreshold] = 0.
            maskbatch[maskbatch > self.maskthreshold] = 1.
            # Mask and fill
            if self.maskfill == 'noise':
                return maskbatch * databatch + (1 - maskbatch) * np.random.uniform(low=0., high=1.,
                                                                                   size=databatch.shape)
            elif self.maskfill == 'zeros':
                return maskbatch * databatch

    # Method to restart generators
    def restartgenerator(self):
        # Sync data and mask generators (this also restarts generators)
        self.datacargo.syncgenerators(self.maskcargo)
        self.batchindexiterator = zip(self.datacargo.batchstream(), self.maskcargo.batchstream())

    # Iterator to stream batches
    def batchstream(self):
        # Get loopin'
        for databatch, maskbatch in self.batchindexiterator:
            # Mask and yield
            yield self.maskbatch(databatch, maskbatch)

    # Define iterator methods to use a masker object as an iterator
    def __iter__(self):
        return self

    # Define next method
    def next(self):
        return self.batchstream().next()


# Class to handle a folder of video frames
class videoframes:
    # Constructor
    def __init__(self, path, framesperbatch=1, dim=3, batchsize=20, preptrain=None):
        """
        Class to stream (sequential) data from a folder of video frames.

        :type path: str
        :param path: Path to the folder.

        :type framesperbatch: int
        :param framesperbatch: Number of frames per datapoint.

        :type dim: int
        :param dim: Data dimensionality. Possible values are 2D and 3D, where 2D yields batches of shape
                    (batchsize, numchannels, y, x) (given framesperbatch = 1) and 3D yields batches of shape
                    (batchsize, framesperbatch, numchannels, y, x).

        :type batchsize: int
        :param batchsize: Batch size

        :type preptrain: prepkit.preptrain
        :param preptrain: Train of preprocessing functions. See prepkit.preptrain for more.

        """

        # Meta
        self.path = path
        self.framesperbatch = framesperbatch
        self.dim = dim
        self.batchsize = batchsize
        self.preptrain = preptrain if preptrain is not None else pk.preptrain([])

        # Build a list of readable files and make sure they're sorted in the correct order.
        self.filenames = sorted(glob.glob(self.path + "/*.png"),
                                key=lambda x: [pyk.try2int(c) for c in re.split('([0-9]+)', x)])

        # Compute the effective number of frames
        self.numframes = len(self.filenames) - (len(self.filenames) % self.batchsize)

        # The video is split in batchsize number of chunks. Say B1 and B2 are two sequential batches of shape
        # (batchsize, framesperbatch, numchannels, y, x). For the sake of simplicity, say framesperbatch = 1, in which
        # case frame B2[3, 0, ...] follows frame B1[3, 0, ...] in the video. One therefore has batchsize videos being
        # processed "in parallel". The one video (specified by self.filenames) must therefore be split in batchsize
        # videos.

        # assert int(self.numframes/self.batchsize) <= self.framesperbatch, "Video sequence length too large. Choose a " \
        #                                                                  "smaller batchsize or framesperbatch."

        # In a video with 300 frames with batchsize 3, the splits will be at frames [0, 100, 200]
        self.batchsplits = [i * int(self.numframes / self.batchsize) for i in range(self.batchsize)]

        # Iteration business
        # Generator (of a list of slice objects)
        self.batchindexiterator = None
        self.restartgenerator()

    def restartgenerator(self):
        # BII is [list of batch-wise slices]
        self.batchindexiterator = iter([[slice(bs + strt, bs + strt + self.framesperbatch) for bs in self.batchsplits]
                                        for strt in range(0, self.numframes / self.batchsize, self.framesperbatch)])

    def batchstream(self):
        # Stream batches
        for batchind in self.batchindexiterator:
            # For RGB: batch.shape = (numbatches, T, row, col, ch)
            # For GrS: batch.shape = (numbatches, T, row, col)
            batch = np.array([np.array([ndi.imread(fname) for fname in self.filenames[slce]])
                              for slce in batchind])

            # If batch is 5D, the 5th dimension must be channel. Reshape to fix that
            if batch.ndim == 5 and self.dim == 3:
                # Show channel axis to its place
                batch = np.rollaxis(batch, 4, 2)
            elif batch.ndim == 4 and self.dim == 3:
                # Add a channel axis
                batch = batch[:, :, np.newaxis, ...]
            elif batch.ndim == 4 and self.dim == 2:
                # Nothing to do
                pass
            elif batch.ndim == 5 and self.dim == 2:
                # If channel dimension is a singleton, squeeze it out
                if batch.shape[-1] == 0:
                    batch = np.squeeze(batch, axis=(4,))
                    # Batch is now of shape (numbatches, T, row, col)
                else:
                    raise NotImplemented("Cannot reshape 5D channeled batch to 2D image.")
            else:
                raise NotImplemented("Image not understood.")

            # Preprocess batch
            pbatch = self.preptrain(batch)
            # return
            yield pbatch

    def next(self):
        return self.batchstream().next()

    def __iter__(self):
        return self


# Class to handle a tracing track (provided as a numpy array of shape (T, numobjects, 2) or as a path to ctrax_
# result.mat file)
class track:
    # Constructor
    def __init__(self, array=None, matfile=None, fieldshape=None, ds=None, eedtgain=1., framesperbatch=1, dim=3,
                 batchsize=20, preptrain=None):
        """
        Class to convert a tracing track (given as a numpy array or a matfile if ctrax2np is available)
        to images with "dots running around".

        :type array: numpy.ndarray
        :param array: Array of tracks. Shape should be (T, numobjects, 2) where the last dimension comprises (x, y)

        :type matfile: str
        :param matfile: Path to the CTRAX (Caltech Multiple Walking Fly Tracker) output .mat file.

        :type fieldshape: tuple or list
        :param fieldshape: Shape of the tracking field.

        :type eedtgain: float
        :param eedtgain: Gain of the exponential euclidian distance transform

        :type framesperbatch: int
        :param framesperbatch:

        :type dim: int
        :param dim: Dimensionality of the output batch (whether 2D or 3D sequential)

        :type batchsize: int
        :param batchsize: Duh.

        :type preptrain: prepkit.preptrain
        :param preptrain: Train of preprocessing functions
        """

        assert not (array is None and matfile is None), "Array or path to a matfile not found."
        assert fieldshape is not None, "Field shape must be given."

        # Load array from matfile if required
        self.array = ndl.ctrax2np(matfile) if array is None else array

        # Meta
        self.fieldshape = list(fieldshape)
        self.framesperbatch = framesperbatch
        self.dim = dim
        self.batchsize = batchsize
        self.preptrain = preptrain if preptrain is not None else pk.preptrain([])
        self.eedtgain = eedtgain
        self.ds = [1, 1] if ds is None else ds

        # Compute the total number of frames
        self.numframes = self.array.shape[0]

        # Default for the object ID's (i.e. track all objects)
        self.objectids = range(self.array.shape[1])

        # The video is split in batchsize number of chunks. Say B1 and B2 are two sequential batches of shape
        # (batchsize, framesperbatch, numchannels, y, x). For the sake of simplicity, say framesperbatch = 1, in which
        # case frame B2[3, 0, ...] follows frame B1[3, 0, ...] in the video. One therefore has batchsize videos being
        # processed "in parallel". The one video (specified by self.filenames) must therefore be split in batchsize
        # videos.

        # assert int(self.numframes / self.batchsize) <= self.framesperbatch, "Video sequence length too large. Choose a " \
        #                                                                     "smaller batchsize or framesperbatch."

        # In a video with 300 frames with batchsize 3, the splits will be at frames [0, 100, 200]
        self.batchsplits = [i * int(self.numframes / self.batchsize) for i in range(self.batchsize)]

        # Iteration business
        # Generator (of a list of slice objects)
        self.batchindexiterator = None
        self.restartgenerator()

    def restartgenerator(self):
        # BII is [list of batch-wise slices]
        self.batchindexiterator = iter([[slice(bs + strt, bs + strt + self.framesperbatch) for bs in self.batchsplits]
                                        for strt in range(0, self.numframes / self.batchsize, self.framesperbatch)])

    def batchstream(self):

        for batchind in self.batchindexiterator:
            # track2batch returns a ndarray of shape (T, row, col), i.e. batch.shape = (batchsize, T, row, col)
            pbatch = np.array([ndl.track2volume(self.array[slce, ...], self.fieldshape, objectids=self.objectids,
                                                edtgain=self.eedtgain, ds=self.ds)
                               for slce in batchind])
            # Reshape to sequential if required
            if self.dim == 3:
                pbatch = pbatch[:, :, np.newaxis, ...]
            # Preprocess batch
            batch = self.preptrain(pbatch)
            # return
            yield batch

    def next(self):
        return self.batchstream().next()

    def __iter__(self):
        return self

    # Specify which objects should be tracked
    def trackobjs(self, *args):
        assert all([arg <= self.array.shape[1] - 1 for arg in args]), "Object not found."
        # Set targets
        self.objectids = args


# Class to handle X, Y datasets.
class tincan:
    # Constructor
    def __init__(self, data, numclasses, batchsize=20, xpreptrain=None, ypreptrain=None, xhowtransform=None,
                 yhowtransform=None, **kwargs):
        """
        Data feeder for generic datasets (MNIST, Cifar-10, Cifar-100).

        :type data: tuple
        :param data: Tuple containing X and Y training data (X, Y)

        :type numclasses: int
        :param numclasses: Number of classes in the dataset (e.g. = 10 for MNIST and Cifar-10, = 100 for Cifar-100)

        :type batchsize: int
        :param batchsize: Batch size.

        :type xpreptrain: prepkit.preptrain
        :param xpreptrain: Train of preprocessing functions on X. See preptrain in prepkit for more on preptrains.

        :type ypreptrain: prepkit.preptrain
        :param ypreptrain: Train of preprocessing functions on Y. Can be set to -1 to channel both X,Y through
                           xpreptrain.

        :type xhowtransform: list or tuple
        :param xhowtransform: how to transform X batch. E.g.:
                              ('b', 1, 's', 's') on a batch of shape (20, 784) --> batch of shape (20, 1, 28, 28)
                              ('b', 3, 's', 's') on a batch of shape (20, 3072) --> batch of shape (20, 3, 32, 32)
                              Keys: 'b' (batch size), 's' (image edge size), 'nc' (number of channels)

        :type yhowtransform: list or tuple
        :param yhowtransform: how to transform Y batch. E.g.:
                              ('b', 'nc', 1, 1) on a batch of shape (20, 10) --> batch of shape (20, 10, 1, 1)
                              Keys: 'b' (batch size), 's' (image edge size), 'nc' (number of channels)

        :keyword preptrain: Same as xpreptrain (there for compatibility with previous version)
        """

        # Compatibility patch
        if "preptrain" in kwargs.keys():
            xpreptrain = kwargs["preptrain"]

        # Meta
        self.batchsize = batchsize
        self.numclasses = numclasses
        self.xpreptrain = xpreptrain if xpreptrain is not None else pk.preptrain([])
        self.ypreptrain = ypreptrain if ypreptrain is not None else pk.preptrain([])
        self.xhowtransform = xhowtransform
        self.yhowtransform = yhowtransform

        # Provided data must be a datastructure like: ((trX, trY), (teX, teY), (vaX, vaY))
        # (tr: training, te: test, va: validation)
        self.data = data

        # Parse data
        # X data is of shape [numimages, numpixels]
        self.X = self.data[0]
        # Y data could be one-hot or not
        if len(data[1].shape) == 1 or data[1].shape[1] == 1:
            # Not yet one-hot. To fix that, reshape data[1] to a vector
            Yc = data[1].reshape((data[1].shape[0],))
            Y = np.zeros(shape=(self.X.shape[0], self.numclasses))
            Y[range(self.X.shape[0]), Yc] = 1.
            self.Y = Y
        elif len(data[1].shape) == 2:
            assert data[1].shape[1] == self.numclasses, \
                "Y data shape is not consistent with the given number of classes..."
            self.Y = data[1]
        else:
            raise NotImplementedError("Y data not understood. Must be a vector or a one-hot coded matrix with the "
                                      "correct number of classes as shape[1].")

        # Build index iterator
        self.batchindexiterator = None
        self.restartgenerator()

    # Method to restart generator. Classid provides which class to generate from
    def restartgenerator(self, classid=None):
        if classid is None:
            imgnums = iter(range(self.Y.shape[0]))
            self.batchindexiterator = it.izip(*[imgnums] * self.batchsize)
        else:
            imgnums = iter(np.argwhere(np.argwhere(self.Y == 1)[:, 1] == classid).squeeze())
            self.batchindexiterator = it.izip(*[imgnums] * self.batchsize)

    # Batchstream
    def batchstream(self):
        # Generator loop
        for inds in self.batchindexiterator:
            # Fetch batches
            if not self.ypreptrain == -1:
                batchX = self.xpreptrain(self.transformbatch(self.X[inds, :], what='X'))
                batchY = self.ypreptrain(self.transformbatch(self.Y[inds, :], what='Y'))
            else:
                batchX, batchY = self.xpreptrain((self.transformbatch(self.X[inds, :], what='X'),
                                                 self.transformbatch(self.Y[inds, :], what='Y')))
            # Yield
            yield batchX, batchY

    # Transform batch
    def transformbatch(self, batch, what='X'):
        assert what is 'X' or what is 'Y', "'what' argument is not understood. Allowed: 'X' and 'Y'"
        # Do nothing if xhowtransform and yhowtransform are not provided
        if (what == 'X' and self.xhowtransform is None) or (what == 'Y' and self.yhowtransform is None):
            return batch

        # Figure out shape to reshape to from xhowtransform
        newshape = [batch.shape[0] if key == 'b' else np.sqrt(batch.shape[1]) if key is 's' else
                    self.numclasses if key == 'nc' else key
                    for key in (self.xhowtransform if what is 'X' else self.yhowtransform)]
        return batch.reshape(*newshape)

    # Define methods to use a mnist object as an iterator
    def __iter__(self):
        return self

    # Define next method
    def next(self):
        return self.batchstream().next()


# Class to convert any given generator to a Antipasti datafeeder (endowed with a restartgenerator() method)
class feeder(object):
    def __init__(self, generator, genargs=None, genkwargs=None, preptrain=None, numworkers=None):
        """
        Convert a given generator to an Antipasti data feeder (endowed with a restartgenerator and batchstream method).

        :type generator: generator
        :param generator: Generator to be converted to a feeder.

        :type genargs: list
        :param genargs: List of arguments generator may take.

        :type genkwargs: dict
        :param genkwargs: Dictionary of keyword arguments generator may take.

        :type preptrain: prepkit.preptrain
        :param preptrain: Train of preprocessing functions.

        :type numworkers: int
        :param numworkers: Number of workers to parallelize the generator over.

        """

        assert callable(generator), "Generator must be callable."

        # Meta + Defaults
        self.generator = generator
        self.genargs = genargs if genargs is not None else []
        self.genkwargs = genkwargs if genkwargs is not None else {}
        self.preptrain = pk.preptrain([]) if preptrain is None else preptrain
        self.numworkers = numworkers

        self.parallelize = (not (self.numworkers is None or self.numworkers == 1)) and self.preptrain is not None

        self.iterator = None
        self.restartgenerator()

    def restartgenerator(self):
        if self.parallelize:
            self.iterator = pk.prepdistribute(self.generator(*self.genargs, **self.genkwargs), self.preptrain,
                                              self.numworkers)
        else:
            self.iterator = self.generator(*self.genargs, **self.genkwargs)

    def batchstream(self):
        # Stream batches after having applied preptrain
        for batch in self.iterator:
            if self.parallelize:
                pass
            else:
                # Apply preptrain only when the generator is not parallelized
                yield self.preptrain(batch)

    def __iter__(self):
        return self

    def next(self):
        return self.batchstream().next()

# Class to zip multiple generators
class feederzip(object):
    """
    Zip multiple generators (with or without a restartgenerator method)
    """
    def __init__(self, gens, preptrain=None):
        """
        :type preptrain: prepkit.preptrain
        :param preptrain: Train of preprocessing functions

        :type gens: list of generators
        :param gens: List of generators to be zipped.

        :return:
        """
        # Meta
        self.gens = gens
        self.preptrain = preptrain if preptrain is not None else pk.preptrain([])

    def batchstream(self):
        # Fetch from all generators and yield
        while True:
            batchlist = [gen.next() for gen in self.gens]
            yield self.preptrain(batchlist)

    def restartgenerator(self):
        # Restart generator where possible
        for gen in self.gens:
            if hasattr(gen, "restartgenerator"):
                gen.restartgenerator()

    # To use feederzip as an iterator
    def __iter__(self):
        return self

    # Next method to mirror batchstream
    def next(self):
        return self.batchstream().next()


# This is the prototype of next generation Antipasti generators.
class feederweave(object):
    def __init__(self, gens, preptrains=None):
        # Meta
        self.gens = gens
        # PreptrainS, because one may have multiple preptrains, one per generator. But it's okay if user provides just
        # one.
        if isinstance(preptrains, (list, tuple)):
            preptrains = list(preptrains * len(self.gens) if len(preptrains) == 1 else preptrains)
            assert len(preptrains) == len(self.gens), "Not enough or too many preptrains for the given number of " \
                                                      "generators."
        elif isinstance(preptrains, pk.preptrain):
            preptrains = [preptrains,] * len(self.gens)

        elif preptrains is None:
            preptrains = [None,] * len(self.gens)

        else:
            raise NotImplementedError("Preptrains must be a preptrain or a tuple of preptrains.")

        self.preptrains = preptrains

        self.iterator = None

    def batchstream(self):
        while True:
            for gen, preptrain in zip(self.gens, self.preptrains):
                try:
                    yield ((lambda x: x) if preptrain is None else preptrain)(gen.next())
                except StopIteration:
                    return

    def restartgenerator(self):
        for gen in self.gens:
            if hasattr(gen, 'restartgenerator'):
                gen.restartgenerator()
        self.iterator = self.batchstream()

    def next(self):
        if self.iterator is None:
            self.restartgenerator()
        return self.iterator.next()

    def __iter__(self):
        return self


def mnist(path=None, batchsize=20, xpreptrain=None, ypreptrain=None, dataset="train", **kwargs):
    """
    Legacy MNIST loader.

    :type path: str
    :param path: Path to MNIST pickle file.

    :type batchsize: int
    :param batchsize: Batch size (no shit sherlock)

    :type xpreptrain: prepkit.preptrain
    :param xpreptrain: Train of preprocessing functions on X. See preptrain's documentation in prepkit.

    :type ypreptrain: prepkit.preptrain
    :param ypreptrain: Train of preprocessing functions on Y. Can be set to -1 to channel X,Y through xpreptrain.

    :type dataset: str
    :param dataset: Which dataset to use ('train', 'test' or 'validate')

    :rtype: tincan
    """

    # Compatibility patch
    if "preptrain" in kwargs.keys():
        xpreptrain = kwargs["preptrain"]

    # Parse path
    if path is None:
        path = "/Users/nasimrahaman/Documents/Python/DeepBrain/Datasets/mnist.pkl"

    # Unpickle data
    data = pkl.load(open(path))

    # Load the correct X and Y data
    assert dataset in ["train", "test", "validate"], "Dataset can be either of the three strings: " \
                                                     "'train', 'test', 'validate'. "
    datindex = 0 if dataset is "train" else 1 if dataset is "test" else 2
    X, Y = data[datindex]

    # Generate MNIST tincan
    return tincan(data=(X, Y), numclasses=10, batchsize=batchsize, xpreptrain=xpreptrain, ypreptrain=ypreptrain,
                  xhowtransform=['b', 1, 's', 's'], yhowtransform=['b', 'nc', 1, 1])


# Old class to process volumetric data from HDF5
class _cargo:
    def __init__(self, h5path=None, pathh5=None, data=None, axistags=None, batchsize=20, nhoodsize=None, ds=None,
                 window=None, stride=None, preload=False, dataslice=None, preptrain=None, shuffleiterator=True):
        """
        :type h5path: str
        :param h5path: Path to h5 file

        :type pathh5: str
        :param pathh5: Path to dataset in h5 file

        :type data: numpy.ndarray
        :param data: Volumetric data (if no HDF5 given)

        :type axistags: str or list
        :param axistags: Specify which dimensions belong to what. For data 3D, axistag='cij' would imply that the first
                         dimension is channel, second and third i and j respectively.
                         Another example: axistag = 'icjk' --> dim0: i, dim1: channel, dim2:j, dim3: k
                         Note: 2D datasets are batched as (batchsize, channel, i = y, j = x)
                               3D datasets are batched as (batchsize, k = z, channel, i = y, j = x)

        :type batchsize: int
        :param batchsize: Batch size

        :type nhoodsize: list
        :param nhoodsize: Size of the sliding window. Use None for a channel dimensions (see doc axistags).

        :type ds: list
        :param ds: Downsampling ratio. Note that downsampling can also be done as preprocessing, but that would
                   require the upsampled data to be loaded to RAM beforehand.

        :type window: list
        :param window: Specify the sliding window to use while iterating over the dataset.
                       Example: for axistag = ijk:
                                ['x', 'x', [1, 4, 8]] makes a sliding window over the ij planes k = 1, 4 and 8.
                                ['x', [3, 5, 1], 'x'] makes a sliding window over the ik planes j = 3, 5 and 1.
                       Example: for axistag = cijk:
                                ['x', 'x', 'x', [1, 2]] makes a sliding window over ij planes k = 1, and 2.
                                [[1, 2], 'x', 'x', 'x'] makes no sliding window (i.e. channel axis ignored).
                       NOTE: Don't worry about 'x' corresponding to a channel dimension.

        :type stride: list
        :param stride: Stride of the sliding window

        :type preload: bool
        :param preload: Whether to preload data to RAM. Proceed with caution for large datasets!

        :type dataslice: tuple of slice
        :param dataslice: How to slice HDF5 data before building the generator. Must be a tuple of slices.
                          Ignores axistags.
                          Example: dataslice = (slice(0, 700), slice(0, 600), slice(0, 500)) would replace the data
                          loaded from the HDF5 file (or the one provided) with data[0:700, 0:600, 0:500].

        :type preptrain: pk.preptrain
        :param preptrain: Train of preprocessing functions

        :type shuffleiterator: bool
        :param shuffleiterator: Whether to shuffle batchstream generator
        """

        # Meta
        self.h5path = h5path
        self.pathh5 = pathh5
        self.batchsize = batchsize
        self.preptrain = preptrain
        self.shuffleiterator = shuffleiterator
        self.rngseed = random.randint(0, 100)

        # Read h5 file and dataset, fetch input dimension if h5path and pathh5 given
        if h5path is not None and h5path is not None:
            h5file = h5.File(h5path)
            if not preload:
                self.data = h5file[pathh5]
            else:
                self.data = h5file[pathh5][:]
        elif data is not None:
            self.data = data
        else:
            raise IOError('Path to or in h5 data not found and no data variable given.')

        # Slice data if required
        if dataslice is not None:
            self.data = self.data[tuple(dataslice)]

        # Parse window
        self.window = (['x'] * len(self.data.shape) if not window else window)
        assert len(self.window) == len(self.data.shape), \
            "Variable window must be a list with len(window) = # data dimensions"

        # Determine data dimensionality
        self.datadim = len(self.data.shape)

        # Parse axis tags
        if axistags is not None:
            assert len(axistags) == self.datadim, "Axis tags must be given for all dimensions."
            assert all(axistag in ['i', 'j', 'k', 'c'] for axistag in axistags) and len(axistags) == len(set(axistags)), \
                "Axis tags may only contain the characters c, i, j and k without duplicates."
            self.axistags = list(axistags)
        elif self.datadim == 2:
            self.axistags = list('ij')
        elif self.datadim == 3:
            self.axistags = list('ijk')
        elif self.datadim == 4:
            self.axistags = list('cijk')
        else:
            raise NotImplementedError("Supported data dimensions: 2, 3 and 4")

        # Parse downsampling ratio
        # Set defaults
        self.ds = ([(4 if windowspec == 'x' else 1) for windowspec in self.window]
                   if not ds else [nds if self.window[n] == 'x' else 1
                                   for n, nds in enumerate(ds)])

        # Check for length/dimensionality inconsistencies
        assert len(self.ds) == len(self.window)

        # Parse nhoodsize
        # Set defaults:
        self.nhoodsize = ([(100 if windowspec == 'x' else 1) for windowspec in self.window]
                          if not nhoodsize else [nnhs if self.window[n] == 'x' else 1
                                                 for n, nnhs in enumerate(nhoodsize)])

        # Check for length/dimensionality inconsistencies
        assert len(self.nhoodsize) == len(self.window)

        # Replace nhoodsize at channel axis with axis length (this is required for the loop to work)
        if 'c' in self.axistags:
            self.nhoodsize[self.axistags.index('c')] = self.data.shape[self.axistags.index('c')]

        # Determine net dimensions from axistag
        self.netdim = len(self.axistags) - self.axistags.count('c') - len(self.window) + self.window.count('x')

        # Parse strides
        self.stride = ([1] * len(self.data.shape) if not stride else stride)

        # Generate batch index generator
        self.batchindexiterator = None
        self.restartgenerator()

    # Generator to stream batches from HDF5 (after applying preprocessing pipeline)
    def batchstream(self):

        # Iterate from batch index iterator
        for batchinds in self.batchindexiterator:
            # Load raw batch from HDF5 or numpy data
            rawbatch = np.array([self.data[tuple([slice(nstart, nstart + nnhs, nds) if not self.axistags[n] == 'c'
                                                  else slice(None, None) for n, nstart, nnhs, nds in
                                                  zip(xrange(len(self.nhoodsize)), starts, self.nhoodsize, self.ds)])]
                                 for starts in batchinds])

            # Squeeze rawbatch to get rid of singleton dimensions which may have arisen from sampling a 2D plane from a
            # 3D dataset and transform to fit NNet's format
            rawbatch = self.transformbatch(rawbatch)

            # Preprocess
            if self.preptrain is not None:
                ppbatch = self.preptrain(rawbatch)
            else:
                ppbatch = rawbatch

            # Stream away
            yield ppbatch

    # Method to transform fetched batch to a format NNet accepts
    def transformbatch(self, batch):

        # Insert a singleton dimension as placeholder for c if no channel axis specified and accordingly reformat. This
        # singleton dimension may or may not be squeezed out later, but that is to be taken care of.
        # axis tags
        if 'c' not in self.axistags:
            # Switch to indicate if 'c' in axistags
            channelnotinaxistags = True
            # Add singleton dimension
            batch = batch[:, None, ...]
            # Format axis tag accordingly
            axistags = (['b'] if 'b' not in self.axistags else []) + ['c'] + self.axistags
        else:
            channelnotinaxistags = False
            # Add b to axis tag if not present already
            axistags = (['b'] if 'b' not in self.axistags else []) + self.axistags

        # Transform 2D
        if self.datadim == 2:
            batch = np.transpose(batch, axes=[axistags.index('b'), axistags.index('c'),
                                              axistags.index('i'), axistags.index('j')])
            # Squeeze batch. This may or may not take out the 'c' dimension, depending on whether it's a singleton
            # dimension; in case it does, we'll add it back in
            # NOTE: Batch dimension is squeezed out if batchsize = 1. Squeeze samplewise.
            batch = np.array([sample.squeeze() for sample in batch])
            if channelnotinaxistags:
                batch = batch[:, None, ...]

        # Transform 3D
        elif self.datadim == 3:
            batch = np.transpose(batch, axes=[axistags.index('b'), axistags.index('k'),
                                              axistags.index('c'), axistags.index('i'), axistags.index('j')])
            # Squeeze batch. This may or may not take out the 'c' dimension, depending on whether it's a singleton
            # dimension; in case it does, we'll add it back in
            # NOTE: Batch dimension is squeezed out if batchsize = 1. Squeeze samplewise.
            batch = np.array([sample.squeeze() for sample in batch])
            if channelnotinaxistags and self.netdim == 3:
                batch = batch[:, :, None, ...]
            elif channelnotinaxistags and self.netdim == 2:
                batch = batch[:, None, ...]

        return batch

    # Method to restart main (iteration) generators
    def restartgenerator(self, rngseed=None):

        # Parse rngseed
        if not rngseed:
            rngseed = self.rngseed

        if not self.shuffleiterator:
            self.batchindexiterator = it.izip(*[it.product(*[xrange(0, self.data.shape[n] - self.nhoodsize[n] + 1,
                                                                    self.stride[n])
                                                             if form == 'x' else form
                                                             for n, form in enumerate(self.window)])] * self.batchsize)
        else:
            # Seed random number generator
            random.seed(rngseed)
            # Generate generator
            self.batchindexiterator = it.izip(
                *[it.product(*[random.sample(range(0, self.data.shape[n] - self.nhoodsize[n] + 1,
                                                   self.stride[n]),
                                             len(range(0, self.data.shape[n] - self.nhoodsize[n] + 1,
                                                       self.stride[n])))
                               if form == 'x' else form
                               for n, form in enumerate(self.window)])] * self.batchsize)

    # Method to clone cargo crate (i.e. cargo with a different dataset)
    def clonecrate(self, h5path=None, pathh5=None, data=None, syncgenerators=False):

        # Parse
        h5path = (self.h5path if not h5path else h5path)
        pathh5 = (self.pathh5 if not pathh5 else pathh5)
        data = (self.data if data is None else data)

        # Check if the file shapes consistent (save for another day)
        pass

        # FIXME: Sync shuffled generators
        # Make new cargo object
        newcargo = cargo(h5path=h5path, pathh5=pathh5, data=data, axistags=self.axistags, batchsize=self.batchsize,
                         nhoodsize=self.nhoodsize, ds=self.ds, window=self.window, stride=self.stride,
                         shuffleiterator=self.shuffleiterator)

        # Sync generators if requested
        if syncgenerators:
            self.syncgenerators(newcargo)

        # return
        return newcargo

    # Method to copy a cargo object
    def copy(self, syncgenerators=True):
        # Make new cargo
        newcargo = cargo(h5path=self.h5path, pathh5=self.pathh5, data=self.data, axistags=self.axistags,
                         batchsize=self.batchsize, nhoodsize=self.nhoodsize, ds=self.ds, window=self.window,
                         stride=self.stride,
                         preptrain=self.preptrain, shuffleiterator=True)

        # sync generators if requested
        if syncgenerators:
            self.syncgenerators(newcargo)

        # and return
        return newcargo

    # Method to synchronize the generators of self and a given cargo instance
    def syncgenerators(self, other):
        """
        :type other: cargo
        """
        # Set rngseeds
        other.rngseed = self.rngseed
        # Restart both generators
        self.restartgenerator()
        other.restartgenerator()

    # Define self as an iterator
    def __iter__(self):
        return self

    # Define next method
    def next(self):
        return self.batchstream().next()
