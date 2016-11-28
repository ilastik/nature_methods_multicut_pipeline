import os
import yaml
import sys

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

nnetpath = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
sys.path.append(os.path.join(nnetpath, "Antipasti"))

import Antipasti.prepkit as pk
import Antipasti.netdatakit as ndk
import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl


def prepfunctions():
    """
    Function to generate preprocessing functions.

    :return: dictionary of preprocessing functions.
    """

    # Function to apply exp-euclidean distance transform
    def disttransform(gain):

        def fun(batch):
            # Invert batch
            batch = 1. - batch
            # Merge batch and channel dimensions
            bshape = batch.shape
            batch = batch.reshape((bshape[0] * bshape[1], bshape[2], bshape[3]))
            # Distance transform by channel
            transbatch = np.array([np.exp(-gain * distance_transform_edt(img)) for img in batch])
            # Reshape batch and return
            return transbatch.reshape(bshape)

        return fun

    # Function to add the complement of a batch as an extra channel
    def catcomplement():
        def _catcomplement(batch):
            # Compute complement
            cbatch = 1. - batch
            return np.concatenate((batch, cbatch), axis=1)
        return _catcomplement

    # Function for combined elastic tranformation
    def elastictransform(sigma, alpha):

        def _elastictransform(batches):
            batch1, batch2 = batches
            rng = np.random.RandomState(None)

            def func(image1, image2):
                assert image1.shape == image2.shape
                # Take measurements
                imshape = image1.shape
                # Make random fields
                dx = rng.uniform(-1, 1, imshape) * alpha
                dy = rng.uniform(-1, 1, imshape) * alpha
                # Smooth dx and dy
                sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
                sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
                # Make meshgrid
                x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
                # Distort meshgrid indices
                distinds = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)
                # Map cooordinates from image to distorted index set
                transformedimage1 = map_coordinates(image1, distinds, mode='reflect').reshape(imshape)
                transformedimage2 = map_coordinates(image2, distinds, mode='reflect').reshape(imshape)
                return transformedimage1, transformedimage2

            # What's done here can be done by synchronizing random number generators, but let's save that can of worms
            # for another day.
            newbatch1, newbatch2 = [], []
            for sample1, sample2 in zip(batch1, batch2):
                newsample1, newsample2 = [], []
                for image1, image2 in zip(sample1, sample2):
                    newimage1, newimage2 = func(image1, image2)
                    newsample1.append(newimage1)
                    newsample2.append(newimage2)
                newsample1, newsample2 = np.array(newsample1), np.array(newsample2)
                newbatch1.append(newsample1)
                newbatch2.append(newsample2)
            newbatch1, newbatch2 = np.array(newbatch1), np.array(newbatch2)

            return newbatch1, newbatch2

        return _elastictransform

    # Function for combined rotation
    def randomrotate(rngseed=None):

        def _randomrotate(batches):
            batch1, batch2 = batches
            rng = np.random.RandomState(rngseed)

            def func(image1, image2):
                k = rng.randint(0, 4)
                return np.rot90(image1, k=k), np.rot90(image2, k=k)

            newbatch1, newbatch2 = [], []
            for sample1, sample2 in zip(batch1, batch2):
                newsample1, newsample2 = [], []
                for image1, image2 in zip(sample1, sample2):
                    newimage1, newimage2 = func(image1, image2)
                    newsample1.append(newimage1)
                    newsample2.append(newimage2)
                newsample1, newsample2 = np.array(newsample1), np.array(newsample2)
                newbatch1.append(newsample1)
                newbatch2.append(newsample2)
            newbatch1, newbatch2 = np.array(newbatch1), np.array(newbatch2)

            return newbatch1, newbatch2

        return _randomrotate

    prepfuncs = {"randomrotate": randomrotate,
                 "elastictransform": elastictransform,
                 "catcomplement": catcomplement,
                 "disttransform": disttransform}

    return prepfuncs


def loaddata(path):
    """
    Function to load in a TIFF or HDF5 volume from file in `path`.

    :type path: str
    :param path: Path to file (must end with .tiff or .h5).
    """
    if path.endswith(".tiff") or path.endswith(".tif"):
        try:
            from vigra.impex import readVolume
        except ImportError:
            raise ImportError("Vigra is needed to read/write TIFF volumes, but could not be imported.")

        volume = readVolume(path)
        return volume

    elif path.endswith(".h5"):
        try:
            from Antipasti.netdatautils import fromh5
        except ImportError:
            raise ImportError("h5py is needed to read/write HDF5 volumes, but could not be imported.")

        volume = fromh5(path)
        return volume

    else:
        raise NotImplementedError("Can't load: unsupported format. Supported formats are .tiff and .h5")


# Save data
def savedata(data, path):
    """
    Saves volume as a .tiff or .h5 file in path.

    :type data: numpy.ndarray
    :param data: Volume to be saved.

    :type path: str
    :param path: Path to the file where the volume is to be saved. Must end with .tiff or .h5.
    """
    if path.endswith(".tiff") or path.endswith('.tif'):
        try:
            from vigra.impex import writeVolume
        except ImportError:
            raise ImportError("Vigra is needed to read/write TIFF volumes, but could not be imported.")

        writeVolume(data, path, '', dtype='UINT8')

    elif path.endswith(".h5"):
        try:
            from vigra.impex import writeHDF5
            vigra_available = True
        except ImportError:
            vigra_available = False
            import h5py

        if vigra_available:
            writeHDF5(data, path, "/data")
        else:
            with h5py.File(path, mode='w') as hf:
                hf.create_dataset(name='data', data=data)

    else:
        raise NotImplementedError("Can't save: unsupported format. Supported formats are .tiff and .h5")


def datafeeders(gtpath, rdpath, batchsize=4):
    """
    Generates Antipasti datafeeders given paths to (ISBI) raw data and ground truth.

    :type gtpath: str
    :param gtpath: Path to ground truth (could be a tiff or a hdf5 file).

    :type rdpath: str
    :param rdpath: Path to raw data (could be a tiff or a hdf5 file).

    :type batchsize: int
    :param batchsize: Batch size of the data feeder. If there's not enough space in your GPU to train the model, this
                      should be the first parameter reduced.
    """

    # Load data
    dY = loaddata(gtpath)
    dX = loaddata(rdpath)

    # Fetch preprocessing functions
    prepfuncs = prepfunctions()

    # Default Training and validation split
    # Training
    trdX, trdY = dX[0:28, ...], dY[0:28, ...]
    # Validation
    vadX, vadY = dX[28:, ...], dY[28:, ...]

    # Check shapes
    assert trdX.shape == (28, 512, 512), "Shape mismatch (raw training volume). Expected {}, got {}.".\
        format((28, 512, 512), trdX.shape)
    assert trdY.shape == (28, 512, 512), "Shape mismatch (training labels). Expected {}, got {}.".\
        format((28, 512, 512), trdY.shape)

    assert vadX.shape == (2, 512, 512), "Shape mismatch (raw validation volume). Expected {}, got {}.". \
        format((2, 512, 512), vadX.shape)
    assert vadY.shape == (2, 512, 512), "Shape mismatch (validation labels). Expected {}, got {}.". \
        format((2, 512, 512), vadY.shape)

    def gen(X, Y):
        assert X.shape == Y.shape, "Raw data and labels must have the same shape. Shape (raw): {}, shape (labels): {}"\
            .format(X.shape, Y.shape)

        # Count the number of images in X
        numimg = X.shape[0]
        # Get a shuffled list of image indices
        idxlist = list(np.random.permutation(range(numimg)))

        while True:
            batchXlist, batchYlist = [], []

            for _ in range(batchsize):
                try:
                    idx = idxlist.pop()
                except IndexError:
                    continue
                # Load images from volume X and Y
                batchXlist.append(X[idx, ...])
                batchYlist.append(Y[idx, ...])

            # If volume exhausted,
            if len(batchXlist) == 0:
                return
                # generator is kill :(

            # Make numpy arrays
            batchX = np.array(batchXlist)[:, np.newaxis, ...]
            batchY = np.array(batchYlist)[:, np.newaxis, ...]

            # Build trains of preprocessing functions (negative exponential of distance transform, random elastic
            # transforms and random rotations)
            prepX = pk.preptrain([pk.normalizebatch(), pk.cast()])
            prepY = pk.preptrain([pk.im2double(8), prepfuncs['disttransform'](0.7), pk.cast()])
            prepXY = pk.preptrain([pk.preptrain([prepfuncs['elastictransform'](50., 2000.),
                                                 prepfuncs['randomrotate'](),
                                                 lambda batches: (batches[0],
                                                                  prepfuncs['catcomplement']()(batches[1]))])])

            # Process batch and yield
            batchXY = prepXY((prepX(batchX), prepY(batchY)))
            yield batchXY

    # Make Antipasti data feeders from generators
    # Training
    tr = ndk.feeder(gen, genargs=[trdX, trdY])
    # Validation
    va = ndk.feeder(gen, genargs=[vadX, vadY])

    return tr, va


# Download pretrained weights
def downloadweights(where=None):
    raise NotImplementedError

# Get paths to the download weight
def getweightpaths(masterpath):
    """
    Utility function to get paths to the weight files given the path to a master folder. This folder must contain three
    subfolders named 'i1', 'i2' and 'i3', each of which must contain the respective weight file (of the respective
    member in the ensemble).

    :type masterpath: str
    :param masterpath: Path to the master weight folder.
    """

    # Get folders in masterpath
    folders = os.listdir(masterpath)
    # Make sure folders match pattern
    assert set(folders) == {'i1', 'i2', 'i3'}, "Masterpath must contain three folders: 'i1', 'i2' and 'i3'."
    # Fetch parameter paths
    parampath = []

    for folder in folders:
        paramfolderpath = masterpath + os.sep + folder
        parampath.append(paramfolderpath + os.sep + os.listdir(paramfolderpath)[-1])

    return parampath

# Build network
def buildmodel(dropout=True, parampath=None):
    """
    Build ICv1 and load in parameters from a weight file in parampath.

    :type dropout: bool
    :param dropout: Whether to use dropout

    :type parampath: str
    :param parampath: Path to the weight file.
    """



    # Define shortcuts
    # Convlayer with ELU
    cl = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                         activation=ntl.elu())

    # Convlayer without activation
    cll = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize)

    # Strided convlayer with ELU (with autopad)
    scl = lambda fmapsin, fmapsout, kersize, padding=None: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout,
                                                                        kersize=kersize,
                                                                        stride=[2, 2], activation=ntl.elu(),
                                                                        padding=padding)
    # Strided 3x3 pool layerlayertrain or Antipasti.netarchs.layertrainyard
    spl = lambda: nk.poollayer(ds=[3, 3], stride=[2, 2], padding=[1, 1])

    # 2x2 Upscale layer
    usl = lambda: nk.upsamplelayer(us=[2, 2])

    # Softmax
    sml = lambda: nk.softmax(dim=2)

    # Identity
    idl = lambda: ak.idlayer()

    # Replicate
    repl = lambda numrep: ak.replicatelayer(numrep)

    # Merge
    merl = lambda numbranch: ak.mergelayer(numbranch)

    # Dropout
    drl = lambda p=(0.5 if dropout else 1.): nk.noiselayer(noisetype='binomial', p=p)

    # Inception module
    def inceptionize(streams):
        # Compute number of streams
        numstreams = len(streams)
        # Multiply
        module = na.layertrainyard([streams])
        # Build replicate and merge layers
        rep = ak.replicatelayer(numstreams)
        mer = ak.mergelayer(numstreams)
        # Build and return inception module
        return rep + module + mer

    # Build the network
    # --- a1 --- b1 --- --- c1 --- d1 --- d2 --- c2 --- --- b1 --- a1 ---
    #                  |                               |
    #                   ------------- id --------------

    a1 = cl(1, 32, [9, 9]) + drl() + cl(32, 48, [9, 9])

    b1 = scl(48, 128, [7, 7]) + drl() + \
         inceptionize([cl(128, 64, [3, 3]) + cl(64, 64, [1, 1]), cl(128, 64, [5, 5]) + cl(64, 64, [3, 3])]) + \
         cl(128, 160, [3, 3])

    c1 = inceptionize([cl(160, 64, [5, 5]) + spl(), scl(160, 64, [3, 3]) + cl(64, 96, [1, 1])]) + \
         cl(160, 160, [3, 3]) + drl() + \
         inceptionize([cl(160, 100, [7, 7]), cl(160, 48, [5, 5]) + cl(48, 48, [1, 1]),
                       cl(160, 64, [3, 3]) + cl(64, 64, [1, 1])]) + \
         cl(212, 240, [3, 3])

    d1 = inceptionize([cl(240, 192, [1, 1]) + spl(), scl(240, 512, [3, 3])]) + cl(704, 1024, [3, 3])

    d2 = drl() + inceptionize([cl(1024, 384, [3, 3]) + cl(384, 200, [3, 3]), cl(1024, 260, [1, 1]),
                               cl(1024, 384, [5, 5]) + cl(384, 200, [1, 1])]) + \
         cl(660, 512, [3, 3]) + \
         inceptionize([cl(512, 60, [7, 7]), cl(512, 180, [3, 3])]) + \
         usl()

    c2 = drl() + cl(240, 200, [3, 3]) + \
         inceptionize([cl(200, 140, [3, 3]) + cl(140, 80, [3, 3]), cl(200, 140, [5, 5]) + cl(140, 80, [5, 5])]) + \
         cl(160, 160, [5, 5]) + \
         usl()

    b2 = drl() + cl(320, 128, [5, 5]) + \
         inceptionize([cl(128, 60, [9, 9]) + cl(60, 48, [5, 5]), cl(128, 72, [5, 5]) + cl(72, 48, [5, 5])]) + \
         cl(96, 60, [5, 5]) + \
         cl(60, 48, [3, 3]) + \
         usl()

    a2 = drl() + cl(48, 32, [9, 9]) + cl(32, 16, [5, 5]) + cl(16, 16, [3, 3]) + cll(16, 2, [1, 1]) + sml()

    # Putting it together
    interceptorv1 = a1 + b1 + repl(2) + (c1 + d1 + d2 + c2) * idl() + merl(2) + b2 + a2
    interceptorv1.feedforward()

    # Load parameterslayertrain or Antipasti.netarchs.layertrainyard
    if parampath is not None:
        interceptorv1.load(parampath)

    return interceptorv1


# Train the model
def trainmodel(model, tr, va, backuppath=None, numepochs=1000, verbosity=5, validationfrequency=25, backupfrequency=50):
    """
    Train ICv1 given training and validation datafeeders.

    Before training:
    Make sure ``backuppath`` is set: this is where the network backs up its parameters. Make sure there's enough disk
    space available to prevent the program from raising an ``IOError``.

    During training:
    The network will train for ``numepoch`` epochs and print the progress at every iteration. The parameters will be
    backed up every ``backupfrequency`` iterations, and validation will occur every ``validationfrequency`` iterations.

    After training:
    If training a single network:

        The parameter set yielding the best validation error is saved as ``ltp-yyyy-mm-dd--hh-mm-ss-best.save`` in
        ``backuppath``. To load in this set of parameters, execute ``model.load(path2file)``, where ``path2file`` is the
         path to the saved parameter file.

    If training an ensemble:

        The parameters yielding the best validation results must be copied to a directory with the following structure:
        paramdir
        |__ i1
        |    |__ weights.save
        |
        |__ i2
        |    |__ weights.save
        |
        |__ i3
             |__ weights.save

        where path to paramdir must be specified in the config .yaml file (as parampath), subdirectories i1, i2 and i3
        contain the best validation weights of the three ensemble members. Finally, to infer with the new weights,
        simply call the function 'ensembleinferisbi' with its arguments.

    :type model: Antipasti.netarchs.model
    :param model: Model to be trained. layertrain or Antipasti.netarchs.layertrainyard

    :type tr: Antipasti.netdatakit.feeder
    :param tr: Antipasti training feeder. To make a training feeder from an arbitrary python generator function
               ``genfunc`` with arguments ``genargs`` and keyword arguments ``genkwargs``, call
               ``Antipasti.netdatakit.feeder(genfunc, genargs, genkwargs)``, where ``genargs`` is a list and
               ``genkwargs`` is a dictionary.

    :type va: Antipasti.netdatakit.feeder
    :param va: Antipasti validation feeder.

    :param backuppath: Where to back up network parameters. This folder could potentially get very large (~1TB for small
                       backupfrequency/validationfrequncy). By default, parameters are saved in weight files, which are
                       named 'ltp-yyyy-mm-dd--hh-mm-ss-best.save' (parameters with the best validation error) or
                       'ltp-yyyy-mm-dd--hh-mm-ss-routine.save' (parameters which are backed up every `backupfrequency`
                       iterations).

    :type numepochs: int
    :param numepochs: Number of epochs to train for.

    :type verbosity: int
    :param verbosity: Verbosity of the trainer.

    :type validationfrequency: int
    :param validationfrequency: How often validation occurs (in number of iterations)

    :type backupfrequency: int
    :param backupfrequency: How often routine backup occurs (in number of iterations)

    :return: Trained model.
    """

    # Assign working directory
    model.savedir = backuppath
    # Build computational graph to compute MSE validation error
    model.error(modeltype='regression')
    # Build computational graph to compute cross entropy loss (with L2 weight decay coefficient = 0.0005)
    model.cost(method='cce', regterms=[(2, 0.0005)])
    # Get optimizer updates
    model.getupdates(method='adam')
    # Fit model
    model.fit(trX=tr, trY=-1, vaX=va, vaY=-1, numepochs=numepochs, verbosity=verbosity,
              validateevery=validationfrequency, progressbarunit=1, backupparams=backupfrequency)

    return model


# Make Inference Function
def inferencefunc(model):
    """
    Compile and build inference function from model with stochastic test time data augmentation. Refer to code for
    data-augmentation parameters.

    :type model: Antipasti.netarchs.model
    :param model: Model to build inference function from. Make sure the correct weights are loaded.

    :return: Inference function
    """
    # Compile inference function
    model.compile(what='inference')

    # Define preprocessing and postprocessing functions (for stochastic data augmentation, i.e. random linear and
    # elastic transformations)
    pt = pk.preptrain([pk.pad(padding=20, mode='reflect', invert=False),
                       pk.randomflip(randomstate=42, invert=False),
                       pk.randomrotate(angle=90, randomstate=42, invert=False),
                       pk.elastictransform(sigma=50., alpha=2500., randomstate=42, invert=False),
                       pk.normalizebatch(),
                       pk.cast()
                       ])

    ipt = pk.preptrain([pk.elastictransform(sigma=50., alpha=2500., randomstate=42, invert=True),
                        pk.randomrotate(angle=90, randomstate=42, invert=True),
                        pk.randomflip(randomstate=42, invert=True),
                        pk.pad(padding=20, invert=True)
                        ])

    # Make inference function
    infun = pk.batch2imagefunc(model.classifier, getprobmap=True, preptrain=pt, posptrain=ipt)

    return infun

# Infer on ISBI
def inferisbi(path, inferfunc, savepath=None, numfolds=20, verbosity=3):
    """
    Infer on ISBI with a given model (specified by ``inferfunc``).
    The path to the ISBI test data must be provided as ``path``.

    :type path: str
    :param path: Path to ISBI test data.

    :type inferfunc: callable
    :param inferfunc: Inference function (generated from ``inferencefunc``)

    :type savepath: str
    :param savepath: Where to save the inferred volume.

    :type numfolds: int
    :param numfolds: Number of times data augmentation should be applied per image.

    :type verbosity: int
    :param verbosity: Verbosity. Must be a number between 1 and 3.

    :return: Inferred volume.
    """
    # Load volume from path
    volume = loaddata(path)

    if verbosity >= 3:
        print "Loaded volume of shape {} from {}.".format(volume.shape, path)

    # Assume volume.shape = (512, 512, 30)
    assert volume.shape == (512, 512, 30), "ISBI test volume has the shape: (512, 512, 30)."

    if verbosity >= 1:
        print "Using function (ID: {}) to infer.".format(id(inferfunc))

    # Transpose volume to a convenient shape (= (30, 512, 512))
    volume = np.rollaxis(volume, 2)
    # Loop over images in volume
    stack = []
    for imagenum in range(volume.shape[0]):
        if verbosity >= 2:
            print "Processing image {} of {}...".format(imagenum, volume.shape[0])
        # Fetch image from volume
        image = volume[imagenum, ...]
        # Run multifold inference
        imagelist = [inferfunc(image) for _ in range(numfolds)]
        # Average results
        image = np.mean(np.array(imagelist), axis=0)
        # Append image to stack
        stack.append(image)

    # Make volume out of stack
    ivolume = np.array(stack)
    # Transpose
    ivolume = np.transpose(ivolume, axes=[1, 2, 0])

    if verbosity >= 3:
        print("Inference done.")

    if savepath is not None:
        if verbosity >= 2:
            print("Saving volume (shape: {}) to {}".format(ivolume.shape, savepath))
        savedata(ivolume, savepath)

    # Return
    return ivolume

def ensembleinferisbi(models, path, savepath=None, numfolds=20, verbosity=3):
    """
    Infer an ensemble of models on ISBI test dataset.

    :type models: list of Antipasti.netarchs.model
    :param models: List of models in the ensemble.

    :type path: str
    :param path: Path to the ISBI test dataset.

    :type savepath: str
    :param savepath: Where to save inferred volume.

    :type numfolds: int
    :param numfolds: Number of times data augmentation should be applied per image.

    :type verbosity: int
    :param verbosity: Verbosity. Must be a number between 1 and 3.

    :return:
    """
    # Fetch inference functions
    infuncs = [inferencefunc(model) for model in models]
    # Infer volumes
    volumes = [inferisbi(path, inferfunc, numfolds=numfolds, verbosity=verbosity) for inferfunc in infuncs]
    # Average volumes
    volume = np.mean(np.array(volumes), axis=0)

    # Save
    if savepath is not None:
        savedata(volume, savepath)

    # Return
    return volume

if __name__ == "__main__":

    # Relative path to the config file
    configpath = os.path.join(nnetpath, 'config/config.yaml')

    with open(configpath, mode='r') as configfile:
        try:
            config = yaml.load(configfile)
        except Exception as e:
            print("Could not parse YAML. The original exception will be raised:")
            raise e


    # Ground Truth PATH
    gtpath = config['gtpath']
    # Raw Data PATH
    rdpath = config['rdpath']
    # Test Raw Data PATH
    trdpath = config['trdpath']
    # Where to save infered data
    infpath = config['infpath']
    # Where to back up network parameters
    backuppath = config['backuppath']
    # Path to parameters for inference
    parampath = config['parampath']
    # Whether to train or to infer
    mode = config['mode']

    assert mode in ['train', 'infer'], "Mode must be 'train' or 'infer'."

    if mode == 'train':
        # Build ICv1
        model = buildmodel(dropout=True)
        # Fetch data feeders
        tr, va = datafeeders(gtpath, rdpath)
        # Train
        trainmodel(model, tr, va, backuppath=backuppath)

    elif mode == 'infer':
        # Get paths to parameters
        ppaths = getweightpaths(parampath)
        # Build ICv1s
        models = [buildmodel(dropout=False, parampath=ppath) for ppath in ppaths]
        # Build ensemble inference function with stochastic data augmentation and all the bells and whistles
        ensembleinferisbi(models, trdpath, infpath, numfolds=20)
    else:
        raise NotImplementedError