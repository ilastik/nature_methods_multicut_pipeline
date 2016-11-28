__author__ = 'nasimrahaman'

__doc__ = \
    """
    PREProcessing KIT: Module to handle preprocessing business.
    Contents:
        preptrain: Train of preprocessing functions.
        prepkit: Collection of basic prep functions. Each of these functions return a callable which goes in as a
                 'coach' in preptrain. coach is only fed 1 argument during feedforward through the train.
    """

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate as _rotate

# Train of PREProcessing functions
class preptrain:
    """Train of preprocessing functions"""
    def __init__(self, train):
        """
        :type train list or callable
        :param train: List of preprocessing functions to apply on the input
        :return:
        """
        # Check inputs
        train = list(train)
        assert all([callable(coach) for coach in train])

        # Meta
        self.train = train

        # Containers for input and output
        self.x = np.array([])
        self.y = np.array([])

    # Method to feed forward
    def __call__(self, inp=None):

        # Parse
        inp = (self.x if inp is None else inp)

        # Instantiate an interloop container
        itc = inp

        # Loop
        for coach in self.train:
            itc = coach(itc)

        # Assign and return
        self.y = itc
        return self.y

    def append(self, fun):
        assert callable(fun), "Appended element must be a callable."
        self.train.append(fun)


# Convenience functions
# Function to convert a given function acting on an image to one that acts on a batch of images
def image2batchfunc(fun, ignorechannels=True):
    """
    Given a callable fun (out image = fun(in image)), batchfunc returns a function that applies fun to all images in a
    training batch.

    :type fun: callable
    :param fun: Image or video function. Must accept either:
                - images of shape (row, col) or (numchannels, row, col) and return an image of similar (but not
                  necessarily same) shape
                - videos of shape (T, row, col) or (numchannels, T, row, col) and return a video of similar (but not
                  necessarily same) shape.

    :type ignorechannels: bool
    :param ignorechannels: Whether fun takes care of the channel axis.
                           If set to false, fun must accept either 2D channeled images of shape
                           (numchannels, row, col) or 3D spatio-temporal images (i.e. videos) of shape
                           (numchannels, T, row, col), depending on the input batch shape.

    :return: Function that applies fun image-wise on a training batch
    """

    def func(batch):
        # Type assertion for Pycharm
        assert isinstance(batch, np.ndarray), "Batch must be a numpy array."
        assert batch.ndim == 5 or batch.ndim == 4, "Batch must be a 4D or 5D numpy.ndarray."

        # Infer dimensionality from batch dimension if dim is not valid
        dim = {5: 3, 4: 2}[len(batch.shape)]

        if dim == 3:
            # batch.shape = (numbatches, T, numchannels, row, col).
            # Reshape to (numbatches, numchannels, T, row, col)
            batch = batch.swapaxes(1, 2)
            # Apply function
            pbatch = np.array([fun(sample) if not ignorechannels else np.array([fun(image) for image in sample])
                               for sample in batch])
            # Reswap axes and return
            pbatch = pbatch.swapaxes(1, 2)
            return pbatch

        elif dim == 2:
            # batch.shape = (numbatches, numchannels, row, col).
            pbatch = np.array([fun(sample) if not ignorechannels else np.array([fun(image) for image in sample])
                               for sample in batch])
            # Return
            return pbatch

    return func


def frame2videofunc(fun):
    """
    Function to convert a function acting on a video frame to one acting on a video frame-wise.

    :type fun: callable
    :param fun: Function to convert. Must take in an image of shape (numchannels, row, col) or (row, col).

    """

    def func(video):
        # Check if video is channeled
        if video.ndim == 3:
            # video.shape = (T, row, col)
            return np.array([fun(frame) for frame in video])
        elif video.ndim == 4:
            # video.shape = (numchannels, T, row, col)
            # Reshape to (T, numchannels, row, col)
            video = video.swapaxes(0, 1)
            # Apply function framewise
            pvideo = np.array([fun(frame) for frame in video])
            # Reshape back to (numchannels, T, row, col)
            return pvideo.swapaxes(0, 1)
        else:
            raise NotImplemented("Input video must be of shape (numchannels, row, col) or (row, col)")

    return func


# Function to convert a function acting on a batch to one on an image. Quite the opposite of image2batchfunc
def batch2imagefunc(fun, ds=None, getprobmap=False, **kwargs):
    """
    Function to convert a function acting on a batch (Inference functions, for instance) to one acting on an image.

    :type fun: Callable
    :param fun: Function to convert. Must accept inputs of shape (numbatches, numchannels, row, col) and return a tensor
                of shape (numbatches, new_numchannels, new_row, new_col)

    :type ds: list or tuple
    :param ds: Shift and stitch downsampling ratio (if applicable)

    :type getprobmap: bool
    :param getprobmap: If your binary classifier model predicts two channels (probability true, probability false),
                       activating this flag results in only the first class being sampled.
                       Say your network outputs a tensor of shape (numbatches, 2, row, col) where
                       output[:, 0, ...] + output[:, 1, ...] = ones_like(output). This flag picks the 0-th class and
                       writes it to the image, i.e. image = batch2image(output[:, 0, ...]).

    :type kwargs: dict
    :param kwargs: Additional arguments.

    :keyword preptrain: Train of preprocessing functions

    :keyword posptrain: Train of postprocessing functions

    :return: Processed image of shape (numchannels, row, col) or (row, col)
    """

    _preptrain = kwargs["preptrain"] if "preptrain" in kwargs.keys() else preptrain([])
    _posptrain = kwargs["posptrain"] if "posptrain" in kwargs.keys() else preptrain([])

    def func(image):
        # Reshape image to a batch
        if image.ndim == 2:
            batch = image[np.newaxis, np.newaxis, ...]
        elif image.ndim == 3:
            batch = image[np.newaxis, ...]
        else:
            raise NotImplemented("Image bust be 2D (row, col) or 3D (channels, row, col)")

        if ds is not None:
            shiftxshape = batch.shape
            # Apply Shift, apply inference function, stitch and return
            shiftedbatch = _preptrain(shiftbatch(ds)(batch))
            pbatch = fun(shiftedbatch)[:, 0:1, ...] if getprobmap else fun(shiftedbatch)
            outbatch = stitchbatch(ds, shiftxshape)(_posptrain(pbatch))
            return outbatch.squeeze()
        else:
            outbatch = fun(_preptrain(batch))[:, 0:1, ...] if getprobmap else fun(_preptrain(batch))
            return _posptrain(outbatch).squeeze()

    return func


# Basic Data Preprocessing
#: TODO: Function to scale data between a given interval (low to high)
def scale2range(low=0, high=0, samplewise=True, featurewise=True):
    pass


# Function to cast image to a given format
def cast(dtype='float32'):
    return lambda X: X.astype(dtype)


#: Function to center a double image (to get it b/w -0.5 to 0.5)
def centerdoubleimage():
    return lambda X: X - 0.5


#: Function to normalize a batch (to mean 0 and variance 1)
def normalizebatch():
    eps = np.finfo(np.float32).eps
    return lambda X: (X - X.mean())/(X.std() + eps)


#: Function to convert n-bit images to double
def im2double(nbit=4):
    return lambda X: X*(1./(2**nbit - 1))


# Function to pad images in a batch
def pad(padding=0, mode='reflect', invert=False):
    def _pad(im):
        assert im.ndim == 2, "Can only pad 2D images."
        # Return if no padding is requested
        if padding == 0:
            return im
        if not invert:
            im = np.pad(im, padding, mode=mode)
        else:
            im = im[padding:(-padding), padding:(-padding)]

        return im

    # Convert image function to batch function and return
    return image2batchfunc(_pad, ignorechannels=True)

#: Function for elastic transformation of images in a batch
def elastictransform(sigma, alpha, randomstate=None, invert=False, padding=0, ignorechannels=True):

    # rng is going to end up in _elastictransform's closure, which should guarantee persistence over function calls
    if isinstance(randomstate, int):
        rng = np.random.RandomState(randomstate)
    elif isinstance(randomstate, np.random.RandomState):
        rng = randomstate
    else:
        rng = np.random.RandomState(None)

    # Define function on image
    def _elastictransform(image):
        assert image.ndim == 2, "Can only transform 2D images."

        # Pad image if required
        if not invert and padding > 0:
            # Pad
            image = np.pad(image, padding, mode='reflect')

        # Take measurements
        imshape = image.shape
        # Make random fields
        dx = rng.uniform(-1, 1, imshape) * alpha
        dy = rng.uniform(-1, 1, imshape) * alpha

        if __debug__ and False:
            print("RNG Debug on _elastictransform: ")
            print("Invert: {}, dx[0, 0]: {}".format(invert, dx[0, 0]))
            print("Invert: {}, dy[0, 0]: {}".format(invert, dy[0, 0]))

        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
        # Distort meshgrid indices (invert if required)
        if not invert:
            distinds = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)
        else:
            distinds = (y - sdy).reshape(-1, 1), (x - sdx).reshape(-1, 1)
        # Map cooordinates from image to distorted index set
        transformedimage = map_coordinates(image, distinds, mode='reflect').reshape(imshape)

        # Crop image if required
        if invert and padding > 0:
            transformedimage= transformedimage[padding:-padding, padding:-padding]

        return transformedimage

    # Convert image function to batch function and return
    return image2batchfunc(_elastictransform, ignorechannels=ignorechannels)


#: Function for random rotations of the image
def randomrotate(angle=90, randomstate=None, invert=False, padding=0, extrapadding=0):
    ignorechannels=True

    if isinstance(randomstate, int):
        rng = np.random.RandomState(randomstate)
    elif isinstance(randomstate, np.random.RandomState):
        rng = randomstate
    else:
        rng = np.random.RandomState(None)

    def _randomrot90(im):

        k = rng.randint(0, 4)
        if not invert:
            if im.ndim == 2:
                return np.rot90(im, k=k)
            elif im.ndim == 3:
                return np.array([np.rot90(frame) for frame in im])
        else:
            if im.ndim == 2:
                return np.rot90(im, k=(4-k))
            elif im.ndim == 3:
                return np.array([np.rot90(frame) for frame in im])

    def _randomrot45(im):
        assert im.shape[0] == im.shape[1], "45 degree rotations are tested only for square images."

        assert im.ndim == 2, "Only 2D images are supported (for 45 degree rotations) at present."

        k = int(rng.choice(a=[0, 1, 3, 5, 7, 8], size=1))
        # Rotation angle (in degrees)
        rotangle = 45 * k

        if not invert:
            if k == 0 or k == 8:
                im = np.pad(im, (padding + extrapadding), mode='reflect') if padding > 0 else im
                return im
            else:
                im = _rotate(im, angle=rotangle, resize=True, mode='reflect')
                im = np.pad(im, padding, mode='reflect') if padding > 0 else im
                return im
        else:
            if k == 0 or k == 8:
                im = im[(padding + extrapadding):-(padding + extrapadding),
                     (padding + extrapadding):-(padding + extrapadding)] if padding > 0 else im
                return im
            else:
                im = im[padding:-padding, padding:-padding] if padding > 0 else im
                # For some reason, _rotate doesn't like if it's values are larger than +1 or smaller than -1.
                # Scale
                scale = np.max(np.abs(im))
                im *= (1./scale)
                # Process
                im = _rotate(im, angle=(360 - rotangle), resize=True, mode='reflect')
                # Rescale
                im *= scale
                # Edges of im are now twice as large as they were in the original image. Crop.
                cropstart = im.shape[0]/4
                cropstop = cropstart * 3
                im = im[cropstart:cropstop, cropstart:cropstop]
                return im

    if angle == 45:
        return image2batchfunc(_randomrot45, ignorechannels=ignorechannels)
    elif angle == 90:
        return image2batchfunc(_randomrot90, ignorechannels=ignorechannels)
    else:
        raise NotImplementedError("Curently implemented rotation angles are 45 and 90 degrees.")

# Function for random flips of the image
def randomflip(randomstate=None, invert=False):
    ignorechannels = True

    if isinstance(randomstate, int):
        rng = np.random.RandomState(randomstate)
    elif isinstance(randomstate, np.random.RandomState):
        rng = randomstate
    else:
        rng = np.random.RandomState(None)

    def _randomflip(im):
        rlr = rng.randint(0, 2)
        rud = rng.randint(0, 2)

        # invert argument is not used because fliplr is its own inverse.
        if rlr == 0:
            if im.ndim == 2:
                im = np.fliplr(im)
            else:
                im = np.array([np.fliplr(frame) for frame in im])
        else:
            im = im

        if rud == 0:
            if im.ndim == 2:
                im = np.flipud(im)
            else:
                im = np.array([np.flipud(frame) for frame in im])
        else:
            im = im

        return im

    return image2batchfunc(_randomflip, ignorechannels=ignorechannels)


#: Shift of shift-and-stitch
def shiftbatch(ds, bordermode="same"):
    # See netkit.shiftlayer.feedforward for more
    def fun(y):
        # Crop y if required
        trimy, trimx = [(dsr - 1)/2 if dsr % 2 == 1 else dsr/2 for dsr in ds]
        y = y[:, :, (trimy):(-trimy), (trimx):(-trimx)] if bordermode is "valid" else y
        # Shift
        dsrat = ds
        npb, fs, ely, elx = y.shape
        return y. \
            reshape((1, npb * fs, ely, elx)). \
            swapaxes(1, 3). \
            reshape((elx / dsrat[1], dsrat[1], ely / dsrat[0], dsrat[0], npb * fs)). \
            swapaxes(0, 3). \
            reshape((dsrat[0] * dsrat[1], ely / dsrat[0], elx / dsrat[1], npb * fs)). \
            swapaxes(1, 2). \
            swapaxes(1, 3). \
            reshape(((dsrat[0] * dsrat[1]) * npb, fs, ely / dsrat[0], elx / dsrat[1]))
    return fun


#: Stitch of shift-and-stitch
def stitchbatch(shiftds, shiftxshape):

    def fun(batch):
        npb = shiftxshape[0]
        # Number of Feature mapS. This might have changed since the shift layer was applied
        fs = batch.shape[1]
        # Edge Length: Y
        ely = shiftxshape[2]
        # Edge Length: X
        elx = shiftxshape[3]
        # DownSampling RATio
        dsrat = shiftds

        # Get stitchin'
        # So this line is basically undo-ing Lukas' line in shift.feedforward
        y = batch. \
            reshape((dsrat[0] * dsrat[1], npb * fs, ely / dsrat[0], elx / dsrat[1])). \
            swapaxes(1, 3). \
            swapaxes(1, 2). \
            reshape((dsrat[0], dsrat[1], ely / dsrat[0], elx / dsrat[1], npb * fs)). \
            swapaxes(0, 3). \
            reshape((1, elx, ely, npb * fs)). \
            swapaxes(1, 3). \
            reshape((npb, fs, ely, elx))

        return y

    return fun


# Function split a batch by frames and apply a function
def smallbatch2batchfunc(fun, splitby='batch', numsplits=2):

    # Parse split by
    if splitby in ['frames', 'frame', 'video', 'sequence', 'time', 'T', 1]:
        axis = 1
    elif splitby in ['batch', 'batches', 0]:
        axis = 0

    def func(batch):
        assert batch.ndim == 5, "Input batch must be 5D Sequential"
        # Trim batch such that it can be split in numsplit chunks
        trim = batch.shape[axis] % numsplits if numsplits != 0 else 0

        if trim != 0:
            batch = batch[:(-trim), ...] if axis == 0 else batch[:, :(-trim), ...]

        if numsplits != 0:
            # Split batch
            batchsplits = np.split(batch, numsplits, axis=axis)
            # Apply function
            outbatch = np.concatenate(([fun(batchsplit) for batchsplit in batchsplits]), axis=axis)
            # Return
            return outbatch
        else:
            # Nothing to split
            return fun(batch)

    return func

if __name__ == "__main__" and __debug__:
    img = np.random.uniform(size=(1, 1, 512, 512))
    pt = preptrain([randomrotate(angle=45, randomstate=42, invert=False, padding=22, extrapadding=106),
                   elastictransform(50., 2500., 42, False),
                   randomflip(randomstate=42, invert=False),
                   randomrotate(angle=90, randomstate=42, invert=False),
                   normalizebatch(),
                   cast()])
    
    ipt = preptrain([randomrotate(angle=90, randomstate=42, invert=True),
                        randomflip(randomstate=42, invert=True),
                        elastictransform(50., 2500., 42, True), 
                        randomrotate(angle=45, randomstate=42, invert=True, padding=22, extrapadding=106)
                       ])
    
    prepinp = np.zeros(shape=(1, 1, 512, 512))
    prepout = pt(prepinp)
    preprec = ipt(prepout)
    
    print("Input shape: {}".format(prepinp.shape))
    print("Output shape: {}".format(prepout.shape))
    print("Recon shape: {}".format(preprec.shape))

    pass