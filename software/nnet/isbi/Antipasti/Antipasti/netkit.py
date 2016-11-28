__author__ = 'Nasim Rahaman'

''' Class definitions for building a modular CNN'''

# DONE: Define classes:
#   convlayer
#   poollayer
#   unpoollayer
#   mlplayer
#   noiselayer

# Global Imports
import theano as th
import theano.tensor as T
import theano.gradient as G
import theano.tensor.nnet.neighbours as nebs
import theano.tensor.nnet.conv as conv
import theano.tensor.nnet.conv3d2d as conv3d2d
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams

from backend import backend
A = backend()

import numpy as np
from warnings import warn
import netutils
import netools
from netarchs import layertrain
from netarchs import layertrainyard
import theanops as tho
import pykit as pyk
import copy as cp

__doc__ = \
    """
    Readme before extending the Antipasti netkit with new layers!

    To define your own layers,
        - Subclass layer.
        - Write at least the feedforward() and inferoutshape() methods.
        - Any parameters you might need go in the _params attribute. DO NOT ASSIGN TO params!
        - You'll need a parameter connectivity map for every parameter. These go in _cparams. Again, DO NOT ASSIGN TO
          cparams!
        - If you do need to know the input shape to define your parameters, consider using ghost variables (ghostvar).
          See the implementation of batchnormlayer for an example.
        - Recurrent layers require a default circuit, which isn't necessarily easy to understand. Get in touch with me
          (drop by my office or shoot me an email (nasim.rahaman@iwr.uni-heidelberg.de)) and we could talk it through.

    The abstract class "layer" does the rest. Don't forget to initialize a layer object from within your layer's
    __init__ method!
    """


# Master class for feedforward layers
class layer(object):
    """ Superclass for all feedforward layers """

    # Constructor
    def __init__(self):
        # Pre-init duck typed parameters
        self.encoderactive = True
        self.decoderactive = True
        self.numinp = 1
        self.numout = 1
        self.inpdim = None
        self.outdim = None
        self.dim = None
        self.allowsequences = False
        self.issequence = False
        self._testmode = False

        self.layerinfo = None

        self._params = []
        self._cparams = []
        self._state = []
        self.updaterequests = []
        self.getghostparamshape = None

        self._circuit = netutils.fflayercircuit()
        self.recurrencenumber = 0.5

        self._inpshape = None
        self.outshape = None
        self.shapelock = False

        self.x = None
        self.y = None
        self.xr = None

    # inputshape as a property
    @property
    def inpshape(self):
        return self._inpshape

    @inpshape.setter
    def inpshape(self, value):
        # Check if shapelock is armed.
        if self.shapelock:
            warn("Can not set input shape. Disarm shapelock and try again.")
            return

        # Get input shape and set outputshape
        self._inpshape = value
        self.outshape = self.inferoutshape(inpshape=value)
        self.outdim = len(self.outshape)

        # Set ghost parameter shapes (if there are any to begin with)
        for param, cparam in zip(self.params, self.cparams):
            if isinstance(param, netutils.ghostvar) and callable(self.getghostparamshape):
                param.shape = self.getghostparamshape(value)

            if isinstance(cparam, netutils.ghostvar) and callable(self.getghostparamshape):
                cparam.shape = self.getghostparamshape(value)


    @property
    def testmode(self):
        return self._testmode

    @testmode.setter
    def testmode(self, value):
        self._testmode = value

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def cparams(self):
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        self._params = value

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        raise AttributeError("Not permitted!")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def isstateful(self):
        return bool(len(self.state))

    # Method to instantiate ghost parameters (if any)
    def instantiate(self):
        """Method to instantiate ghost parameters (if any)."""
        self._params = [param.instantiate() if isinstance(param, netutils.ghostvar) and param.instantiable else
                        param for param in self._params]
        self._cparams = [cparam.instantiate() if isinstance(cparam, netutils.ghostvar) and cparam.instantiable else
                         cparam for cparam in self._cparams]
        self._state = [state.instantiate() if isinstance(state, netutils.ghostvar) and state.instantiable else
                       state for state in self._state]

    # Step method (same as feedforward)
    def step(self, inp):
        """Method to step through a unit in time. Useful only for recurrent layers."""
        return self.feedforward(inp=inp)

    # Feedforward method
    def feedforward(self, inp=None):
        """This method builds the actual theano graph linking input (self.x) to output (self.y)."""
        if inp is None:
            inp = self.x
        return inp

    # Decoder feedforward method
    def decoderfeedforward(self, inp=None):
        """This method builds the theano graph linking output (self.y) to reconstruction (self.xr)."""
        if inp is None:
            inp = self.y
        return inp

    # Apply parameters
    def applyparams(self, params=None, cparams=None):
        """This method applies numerical (or theano shared) parameters to the layer."""
        # Generic method for applying parameters
        if params is not None:
            # Convert to numeric (in case params is symbolic)
            params = netutils.sym2num(params)
            # Loop over all params, and set values
            for param, value in zip(self.params, params):
                param.set_value(value)

        if cparams is not None:
            # Convert to numeric
            cparams = netutils.sym2num(cparams)
            # Loop over all cparams and set values
            for cparam, value in zip(self.cparams, cparams):
                cparam.set_value(value)

    # Method to activate encoder or decoder
    def activate(self, what='all'):
        """Use this method to activate the encoder and/or decoder."""
        if what == 'enc' or what == 'all':
            self.encoderactive = True

        if what == 'dec' or what == 'all':
            self.decoderactive = True

    # Method to deactivate encoder or decoder
    def deactivate(self, what='all'):
        """Use this method to deactivate the encoder and/or decoder."""
        if what == 'enc' or what == 'all':
            self.encoderactive = False

        if what == 'dec' or what == 'all':
            self.decoderactive = False

    # Method for infering output shapes
    def inferoutshape(self, inpshape=None, checkinput=False):
        """Infer the output shape given an input shape. Required for automatic shape inference."""
        if inpshape is None:
            inpshape = self.inpshape
        return inpshape

    # Method to infer state shape
    def inferstateshape(self, inpshape=None):
        # Check if layer is stateful
        if self.isstateful:
            raise NotImplementedError("State shape inference not defined yet.")
        else:
            raise NotImplementedError("Layer is stateless.")

    def __add__(self, other):
        """Stack layers to build a network."""
        # Make sure the number of inputs/outputs check out
        assert self.numout == other.numinp, "Cannot chain a component with {} output(s) " \
                                            "with one with {} input(s)".format(self.numout, other.numinp)

        if isinstance(other, layertrain):
            # Make a layertrain only if chain is linear (i.e. no branches)
            # other.numout = 1 for other a layertrain
            if self.numinp > 1:
                return layertrainyard([self, other])
            else:
                return layertrain([self] + other.train)
        elif isinstance(other, layer):
            # Make a layertrain only if chain is linear (i.e. no branches)
            if all([num == 1 for num in [self.numinp, self.numout, other.numinp, other.numout]]):
                return layertrain([self] + [other])
            else:
                return layertrainyard([self, other])
        elif isinstance(other, layertrainyard):
            return layertrainyard([self] + other.trainyard)
        else:
            raise TypeError('Unrecognized layer class.')

    def __mul__(self, other):
        if isinstance(other, layertrain):
            return layertrainyard([[self, other]])
        elif isinstance(other, layer):
            return layertrainyard([[self, other]])
        elif isinstance(other, layertrainyard):
            return layertrainyard([[self, other]])
        else:
            raise TypeError('Unrecognized layer class.')

    def __div__(self, other):
        raise NotImplementedError("Div method not implemented...")

    def __pow__(self, power, modulo=None):
        raise NotImplementedError("Pow method not implemented...")

    def __repr__(self):
        # Layer spec string
        layerspec = "[{}]".format(self.layerinfo) if self.layerinfo is not None else ""

        desc = "--{}>>> {}{} >>>{}--".format(self.inpshape,
                                             self.__class__.__name__ + "({})".format(str(id(self))),
                                             layerspec,
                                             self.outshape)
        return desc


class convlayer(layer):
    """ Convolutional Layer """

    # Constructor
    def __init__(self, fmapsin, fmapsout, kersize, stride=None, padding=None, activation=netools.linear(), alpha=None,
                 makedecoder=False, zerobias=False, tiedbiases=True, convmode='same', allowsequences=True,
                 inpshape=None, W=None, b=None, bp=None, Wc=None, bc=None, bpc=None, Wgc=None, bgc=None, bpgc=None,
                 allowgradmask=False):

        """
        :type fmapsin: int
        :param fmapsin: Number of input feature maps

        :type fmapsout: int
        :param fmapsout: Number of output feature maps

        :type kersize: tuple or list
        :param kersize: Size of the convolution kernel (y, x, z); A 2-tuple (3-tuple) initializes a 2D (3D) conv. layer

        :type stride: tuple or list
        :param stride: Convolution strides. Must be a 2-tuple (3-tuple) for a 2D (3D) conv. layer.
                       Defaults to (1, ..., 1).

        :type activation: dict or callable
        :param activation: Transfer function of the layer.
                           Can also be a dict with keys ("function", "extrargs", "train") where:
                                function: elementwise theano function
                                trainables: extra parameter variables (for PReLU, for instance)
                                ctrainables: extra parameter connectivity variables

        :type alpha: float
        :param alpha: Initialization gain (W ~ alpha * N(0, 1))

        :type makedecoder: bool
        :param makedecoder: Boolean switch for initializing decoder biases

        :type zerobias: bool
        :param zerobias: Whether not to use bias. True => no bias used (also not included in params).

        :type tiedbiases: bool
        :param tiedbiases: Decoder bias = - Encoder bias when set to True

        :type convmode: str
        :param convmode: Convolution mode. Possible values: 'same' (default), 'full' or 'valid'

        :type allowsequences: bool
        :param allowsequences: Whether to process 3D data as sequences. When set to true and the kernel looks something
                               like [ky, kx, 1] (i.e. kernel[2] = 1), the 3D data is processed by 2D operations
                               (2D convolution).

        :type inpshape: tuple or list
        :param inpshape: Input shapes to expect. Used for optimizing convolutions and recurrent chains.

        :type W: theano tensor of size (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) in 3D,
                                       (fmapsout, fmapsin, kersize[0], kersize[1]) in 2D
        :param W: Preset weight tensor of the layer (use for tying weights)

        :type b: theano vector of size (fmapsout,)
        :param b: Preset bias vector of the layer (use for tying weights)

        :type bp: theano vector of size (fmapsin,)
        :param bp: Preset bias vector of the associated decoder layer (use for tying weights)

        :type Wc: Floated boolean theano tensor of shape identical to that of W
        :param Wc: Connectivity mask of the weight tensor. For a tensor element set to zero, the corresponding element
                   in the weight tensor never gets updated. This can be exploited to have and train two parallel
                   'sublayers' without one interfering with the other.


        :type bc: Floated boolean theano tensor of shape identical to that of b
        :param bc: Connectivity mask of the bias vector. For more documentation see: Wc.

        :type bpc: Floated boolean theano tensor of shape identical to that of bpc
        :param bpc: Connectivity mask of the decoder bias vector. For more documentation see: Wc.

        :type allowgradmask: bool
        :param allowgradmask: Whether to allow gradient masking. There's no reason not to as such, except when using a
                              recurrent layer, the gradient computation fails (known problem).

        :return: None
        """

        # Initialize super class
        super(convlayer, self).__init__()

        # Meta
        self.fmapsin = int(fmapsin)
        self.fmapsout = int(fmapsout)
        self.kersize = list(kersize)
        self.decoderactive = bool(makedecoder)
        self.encoderactive = True
        self.zerobias = bool(zerobias)
        self.tiedbiases = bool(tiedbiases)
        self.convmode = str(convmode)
        self.allowsequences = bool(allowsequences)
        self.allowgradmask = allowgradmask

        # Parse activation
        if isinstance(activation, dict):
            self.activation = activation["function"]
            self.extratrainables = activation["trainables"] if "trainables" in activation.keys() else []
            self.extractrainables = activation["ctrainables"] if "ctrainables" in activation.keys() else \
                [netutils.getshared(like=trainable, value=1.) for trainable in self.extratrainables]
        elif callable(activation):
            self.activation = activation
            self.extratrainables = []
            self.extractrainables = []
        else:
            self.activation = netools.linear()
            self.extratrainables = []
            self.extractrainables = []

        # Name extra trainables for convenience
        for trainablenum, trainable in enumerate(self.extratrainables):
            trainable.name = trainable.name + "-trainable{}:".format(trainablenum) + str(id(self)) \
                if trainable.name is not None else "trainable{}:".format(trainablenum) + str(id(self))
        for trainablenum, trainable in enumerate(self.extractrainables):
            trainable.name = trainable.name + "-ctrainable{}:".format(trainablenum) + str(id(self)) \
                if trainable.name is not None else "ctrainable{}:".format(trainablenum) + str(id(self))

        # Debug Params
        # Encoder and Decoder Convolution Outputs
        self.eIW = None
        self.dIW = None
        # Encoder and Decoder Preactivations
        self.ePA = None
        self.dPA = None

        # Parse initialization alpha
        if alpha is None:
            self.alpha = 1.
            # self.alpha = np.sqrt(1. / (fmapsin * np.prod(kersize)))
        else:
            self.alpha = alpha

        # Parse network dimension (and whether the input a sequence)
        self.issequence = len(self.kersize) == 3 and self.kersize[2] == 1 and self.allowsequences
        self.dim = (2 if len(self.kersize) == 2 or (self.issequence and self.allowsequences) else 3)

        if self.dim == 2:
            self.inpdim = (4 if not self.issequence else 5)
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimension: {}. Supported: 2D and 3D.'.format(self.dim))

        # Parse convolution strides
        if stride is None:
            self.stride = [1, ] * (self.dim + (0 if not self.issequence else 1))
        else:
            if self.dim == 2:
                assert len(stride) == len(self.kersize), "Stride and kersize must have the same length."
                self.stride = list(stride)
            else:
                warn("Convolution strides are presently not supported for 3D convolutions.")
                self.stride = [1, ] * (self.dim + (0 if not self.issequence else 1))

        # Parse padding
        if padding is None:
            # Check if convolution is strided, convmode is 'same' but no padding is provided
            if not all([st == 1 for st in self.stride]) and self.convmode is 'same':
                # Try to infer padding for stride 2 convolutions with odd kersize
                if self.stride == [2, 2] and all([ks % 2 == 1 for ks in self.kersize]):
                    self.padding = [[(ks - 1)/2] * 2 for ks in self.kersize]
                    self.convmode = 'valid'
                else:
                    raise NotImplementedError("Padding could not be inferred for the strided convolution in the 'same' "
                                              "mode. Please provide manually. ")
            else:
                self.padding = [[0, 0], ] * {4: 2, 5: 3}[self.inpdim]
        else:
            assert len(padding) == {4: 2, 5: 3}[self.inpdim], "Padding must be a 3-tuple for 3D or 2D sequential data, " \
                                                              "2-tuple for 2D data."
            # Padding must be [[padleft, padright], ...]
            padding = [[padval, padval] if isinstance(padval, int) else padval[0:2] for padval in padding]
            self.padding = padding
            # Change convmode to valid with a warning
            if not all([st == 1 for st in self.stride]) and self.convmode is 'same':
                warn("Convlayer will apply 'valid' strided convolution to the padded input.")
                self.convmode = 'valid'

        # Initialize weights W and biases b:
        # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
        # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]
        # b.shape = (fmapsout,)

        # Weights, assoc. connectivity mask and gradient clips
        if W is None:
            # Fetch default init scheme (xavier)
            initscheme = netools.xavier
            if self.dim == 3:
                # self.W = th.shared(
                #     value=np.asarray(
                #         self.alpha * np.random.normal(loc=0.0, scale=1.0,
                #                                       size=(fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])),
                #         dtype=th.config.floatX),
                #     name='convW:' + str(id(self)))
                self.W = th.shared(
                    value=self.alpha * initscheme(shape=(fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])),
                    name='convW:' + str(id(self)))
            else:
                # self.W = th.shared(
                #     value=np.asarray(
                #         self.alpha * np.random.normal(loc=0.0, scale=1.0,
                #                                       size=(fmapsout, fmapsin, kersize[0], kersize[1])),
                #         dtype=th.config.floatX),
                #     name='convW:' + str(id(self)))
                self.W = th.shared(
                    value=self.alpha * initscheme(shape=(fmapsout, fmapsin, kersize[0], kersize[1])),
                    name='convW:' + str(id(self)))
        elif isinstance(W, str):
            if W in ["id", "identity"]:
                if self.dim == 2:
                    self.W = netools.idkernel([fmapsout, fmapsin, kersize[0], kersize[1]])
                else:
                    self.W = netools.idkernel([fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]])
            else:
                # Parse init scheme
                initscheme = netutils.smartfunc(getattr(netools, W), ignorekwargssilently=True)
                # Parse kernel shape
                kershape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) if self.dim == 3 else \
                    (fmapsout, fmapsin, kersize[0], kersize[1])
                # Initialize
                self.W = th.shared(value=self.alpha * initscheme(shape=kershape, dtype=th.config.floatX))

            self.W.name = 'convW:' + str(id(self))
        elif callable(W):
            # Parse init scheme
            initscheme = netutils.smartfunc(W, ignorekwargssilently=True)
            # Parse kernel shape
            kershape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) if self.dim == 3 else \
                (fmapsout, fmapsin, kersize[0], kersize[1])
            # Initialize
            self.W = th.shared(value=self.alpha * initscheme(shape=kershape, dtype=th.config.floatX),
                               name='convW:' + str(id(self)))
        else:
            # W must be a shared variable.
            assert netutils.isshared(W), "W must be a shared variable."
            # W must have the right shape!
            Wshape = W.get_value().shape
            kershape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1]) if self.dim == 3 else \
                (fmapsout, fmapsin, kersize[0], kersize[1])
            assert Wshape == kershape, "W is of the wrong shape. Expected shape: {}".format(kershape)

            self.W = W
            self.W.name = 'convW:' + str(id(self))

        # Conn. mask
        if Wc is None:
            self.Wc = netutils.getshared(value=1., like=self.W)
            self.Wc.name = 'convWc:' + str(id(self))
        else:
            assert bool(T.all(T.eq(Wc.shape, self.W.shape)).eval()), \
                "W connectivity mask and W tensor must have the same shape."
            self.Wc = th.shared(T.cast(T.switch(Wc > 0, 1., 0.), dtype='floatX').eval())
            self.Wc.name = 'convWc:' + str(id(self))

        # Gradient clips
        if Wgc is None:
            self.Wgc = [-np.inf, np.inf]
        else:
            assert isinstance(Wgc, (list, np.ndarray)) and len(Wgc) == 2, "Weight filter gradient clips (Wgc) must " \
                                                                          "be a list with two elements."
            self.Wgc = Wgc

        # Biases and conn. mask
        if b is None:
            if not self.zerobias:
                self.b = th.shared(
                    value=self.alpha * np.asarray(np.zeros(shape=(fmapsout,)), dtype=th.config.floatX),
                    name='convb:' + str(id(self)))
            else:
                self.b = th.shared(
                    value=np.asarray(np.zeros(shape=(fmapsout,)), dtype=th.config.floatX),
                    name='convb:' + str(id(self)))
        else:
            self.b = b
            self.b.name = 'convb:' + str(id(self))

        # Conn. mask
        if bc is None and not self.zerobias:
            self.bc = netutils.getshared(value=1., like=self.b)
            self.bc.name = 'convbc:' + str(id(self))
        elif self.zerobias and b is None:
            self.bc = T.zeros_like(self.b)
            self.bc.name = 'convbc:' + str(id(self))
        else:
            assert bool(T.all(T.eq(bc.shape, self.b.shape)).eval()), \
                "b connectivity mask and b vector must have the same shape."
            self.bc = th.shared(T.cast(T.switch(bc > 0, 1., 0.), dtype='floatX').eval())
            self.bc.name = 'convbc:' + str(id(self))

        # Gradient clips
        if not bgc and not self.zerobias:
            self.bgc = [-np.inf, np.inf]
        elif self.zerobias and bgc is None:
            self.bgc = [0, 0]
        else:
            assert isinstance(bgc, (list, np.ndarray)) and len(bgc) == 2, "Bias gradient clips (bgc) must " \
                                                                          "be a list with two elements."
            self.bgc = bgc

        # Decoder Biases and conn. mask
        if not self.tiedbiases:
            if not bp:
                # self.bp = -self.b
                # DEBUG
                self.bp = th.shared(
                    value=self.alpha * np.asarray(np.random.normal(0, 1, size=(fmapsin,)), dtype=th.config.floatX),
                    name='convbp:' + str(id(self)))
            else:
                self.bp = bp
        else:
            self.bp = th.shared(-T.tile(T.mean(self.b, keepdims=True), [fmapsin]).eval())
            self.bp.name = 'convbp:' + str(id(self))

        # Conn. mask
        if not bpc:
            self.bpc = netutils.getshared(value=1., like=self.bp)
            self.bpc.name = 'convbpc:' + str(id(self))
        else:
            assert bool(T.all(T.eq(bpc.shape, self.bp.shape)).eval()), \
                "bp connectivity mask and bp vector must have the same shape."
            self.bpc = th.shared(T.cast(T.switch(bpc > 0, 1., 0.), dtype='floatX').eval())
            self.bpc.name = 'convbpc:' + str(id(self))

        # Gradient clips
        if not bpgc:
            self.bpgc = [-np.inf, np.inf]
        else:
            assert isinstance(bpgc, (list, np.ndarray)) and len(bpgc) == 2, "Decoder bias gradient clips (bpgc) must " \
                                                                            "be a list with two elements."
            self.bpgc = bpgc

        # Fold Parameters
        self._params = [self.W] + ([self.b] if not self.zerobias else []) + \
                       ([self.bp] if self.decoderactive and not self.tiedbiases else []) + self.extratrainables

        self._cparams = [self.Wc] + ([self.bc] if not self.zerobias else []) + \
                        ([self.bpc] if self.decoderactive and not self.tiedbiases else []) + self.extractrainables

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Input shape must have exactly as many elements as input dimension."
            self.inpshape = inpshape

        # Parse output shape
        self.outshape = self.inferoutshape()

        # Container for input (see feedforward() for input shapes) and output
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))
        # Reconstructed input
        self.xr = T.tensor('floatX', [False, ] * self.inpdim, name='xr:' + str(id(self)))

        self.layerinfo = "[{}-in >> {}-out w/ {} kernel]".format(fmapsin, fmapsout, kersize)

    # Params and cparams property definitions
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        # Piggy back on applyparams for back-compatibility
        self.applyparams(params=value)

    @property
    def cparams(self):
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        self.applyparams(cparams=value)

    # Feed forward through the layer
    def feedforward(self, inp=None, activation=None):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        if not self.encoderactive:
            self.y = inp
            return inp

        if not activation:
            activation = self.activation

        __old__ = True

        if not __old__:
            # Check if gradient masking is required
            filtergradmask = self.Wc if self.allowgradmask else None
            biasgradmask = self.bc if self.allowgradmask else None

            # Get PreActivation
            PA = A.conv(inp, self.W, stride=self.stride, padding=self.padding, bias=self.b, filtergradmask=filtergradmask,
                        biasgradmask=biasgradmask, filtergradclips=self.Wgc, biasgradclips=self.bgc, dim=self.dim,
                        convmode=self.convmode, issequence=self.issequence)
            # Apply activation function
            self.y = activation(PA)
            # Return
            return self.y

        if __old__:
            # Get status on bordermodes and co.
            dnnpaddable, bordermode, trim = self._bordermode()

            # Pad with Antipasti if DNN can't do it
            if not dnnpaddable:
                inp = netutils.pad(inp, self.padding)

            # Reshape sequence data
            # Check if input really is sequential
            reallyissequentual = self.issequence and inp.ndim == 5
            # Log initial shape
            inpshape = inp.shape
            if self.issequence:
                if reallyissequentual:
                    assert self.convmode == 'same', "Only 'same' convolutions are supported for sequential data."
                    # This makes inp.shape = (batchnum * T, fmapsin, y, x) out of (batchnum, T, fmapsin, y, x)
                    inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)
                else:
                    warn("convlayer expected a 5D sequential input, but got 4D non-sequential instead.")

            # Apply gradient mask if requested
            if self.allowgradmask:
                # Masked Weight mW
                mW = tho.maskgradient(self.W, self.Wc)
                # Masked bias
                mb = tho.maskgradient(self.b, self.bc)
            else:
                mW = self.W
                mb = self.b

            # Convolve
            if self.dim == 2:
                # IW.shape = (numimages, fmapsout, y, x)
                IW = T.nnet.conv2d(input=inp, filters=G.grad_clip(mW, *self.Wgc),
                                   border_mode=tuple(bordermode) if isinstance(bordermode, list) else bordermode,
                                   filter_shape=self.W.get_value().shape, subsample=tuple(self.stride))

                self.eIW = IW
                PA = IW + G.grad_clip(mb, *self.bgc).dimshuffle('x', 0, 'x', 'x')
                self.ePA = PA
                y = (self._trim(activation(PA), numconvs=1) if trim else activation(PA))

                # Reshape sequence data
                if self.issequence and reallyissequentual:
                    self.y = y.reshape((inpshape[0], inpshape[1], self.fmapsout, inpshape[3], inpshape[4]), ndim=5)
                else:
                    self.y = y

                return self.y

            elif self.dim == 3:
                raise NotImplementedError("3D Convolution is down for maintainance.")
                # IW.shape = (numstacks, z, fmapsout, y, x)
                IW = conv3d2d.conv3d(signals=inp, filters=G.grad_clip(mW, *self.Wgc),
                                     border_mode=('full' if self.convmode is not 'valid' else 'valid'),
                                     signals_shape=tuple(self.outshape) if None not in self.outshape else None,
                                     filters_shape=self.W.get_value().shape)
                self.eIW = IW
                PA = IW + G.grad_clip(mb, *self.bgc).dimshuffle('x', 'x', 0, 'x', 'x')
                self.ePA = PA
                self.y = (self._trim(activation(PA), numconvs=1) if self.convmode is 'same' else activation(PA))
                return self.y

    # Feed forward through the decoder layer (relevant only when used with convolutional autoencoder) [EXPERIMENTAL]
    def decoderfeedforward(self, inp=None, reshape=False, activation=None):
        # Argument inp is expected of the form:
        #   inp.shape = (numimages, z, fmapsout, y, x)      [3D]
        #   inp.shape = (numimages, fmapsout, y, x)         [2D]
        # This layer tries to map a (numimages, z, fmapsout, y, x) image to (numimages, z, fmapsin, y, x).
        # The convolution filters are flipped along the zyx axes.

        # Parse input
        if inp is None:
            inp = self.y
        else:
            self.y = inp

        # Return input if decoder not active
        if not self.decoderactive:
            self.xr = inp
            return inp

        if not activation:
            activation = self.activation

        # Reshape sequence data
        # Check if input really is sequential
        reallyissequentual = self.issequence and inp.ndim == 5
        # Log initial shape
        inpshape = inp.shape
        if self.issequence:
            if reallyissequentual:
                # This makes inp.shape = (batchnum * T, fmapsin, y, x) out of (batchnum, T, fmapsin, y, x)
                inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)
            else:
                warn("convlayer expected a 5D sequential input, but got 4D non-sequential instead.")

        # Reshape if requested
        if self.dim == 3 and reshape:
            inp = inp.dimshuffle(0, 2, 3, 4, 1)

        # Flip conv. kernel and conn. mask
        Wt = self._flipconvfilter()
        Wtc = self._flipconvfilter(self.Wc)

        # Apply gradient mask if requested
        if self.allowgradmask:
            # Masked Weight mW
            mWt = tho.maskgradient(Wt, Wtc)
            # Masked bias
            mbp = tho.maskgradient(self.bp, self.bpc)
        else:
            mWt = Wt
            mbp = self.bp

        if self.dim == 2:
            # Convolve, transfer and return
            # IW.shape = (numimages, fmapsin, y, x)
            IW = conv.conv2d(input=inp, filters=G.grad_clip(mWt, *self.Wgc),
                             border_mode=('full' if self.convmode is not 'valid' else 'valid'))
            self.dIW = IW
            PA = IW + G.grad_clip(mbp, *self.bpgc).dimshuffle('x', 0, 'x', 'x')
            self.dPA = PA
            xr = (self._trim(activation(PA), numconvs=1) if self.convmode is 'same' else activation(PA))

            # Reshape sequence data
            if self.issequence and reallyissequentual:
                # Remember inpshape = (nb, T, nc, r, c)
                self.xr = xr.reshape((inpshape[0], inpshape[1], self.fmapsout, inpshape[3], inpshape[4]), ndim=5)
            else:
                self.xr = xr

            return self.xr

        elif self.dim == 3:
            # Convolve, transfer and return
            # IW.shape = (numstacks, z, fmapsin, y, x)
            IW = conv3d2d.conv3d(signals=inp, filters=G.grad_clip(mWt, *self.Wgc),
                                 border_mode=('full' if self.convmode is not 'valid' else 'valid'))
            self.dIW = IW
            PA = IW + G.grad_clip(mbp, *self.bpgc).dimshuffle('x', 'x', 0, 'x', 'x')
            self.dPA = PA
            self.xr = (self._trim(activation(PA), numconvs=1) if self.convmode is 'same' else activation(PA))
            return self.xr

    # Method to flip conv. filters for the decoder layer
    def _flipconvfilter(self, W=None):
        # Flips the convolution filter along zyx axes and shuffles input and output dimensions.
        # Remember,
        # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
        # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]

        if not W:
            W = self.W

        if self.dim == 2:
            # Shuffle input with output
            Wt = W.dimshuffle(1, 0, 2, 3)
            # Flip yx
            Wt = Wt[::, ::, ::-1, ::-1]
            return Wt

        elif self.dim == 3:
            # Shuffle input with output
            Wt = W.dimshuffle(2, 1, 0, 3, 4)
            # Flip zyx
            Wt = Wt[::, ::-1, ::, ::-1, ::-1]
            return Wt

    # Method to determine cudnn border_mode
    def _bordermode(self):
        # Logic to find what bordermode goes in to the conv interface
        # Find out if padding is compatible with DNN
        dnnpaddable = all([all([dimpad == pad[0] for dimpad in pad]) for pad in self.padding])
        # Compute the dnn pad value
        if dnnpaddable:
            dnnpad = [pad[0] for pad in self.padding]
        else:
            dnnpad = None

        # Whether to trim after conv
        trim = False

        # Get bordermode if padding is [0, 0]
        if dnnpad == [0, 0]:
            if self.convmode == 'same':
                if all([ks % 2 == 1 for ks in self.kersize]):
                    bordermode = 'half'
                else:
                    bordermode = 'full'
                    trim = True
            elif self.convmode == 'valid':
                bordermode = 'valid'
            else:
                bordermode = 'full'
        elif dnnpad is None:
            if self.convmode == 'same':
                bordermode = 'full'
                trim = True
            elif self.convmode == 'valid':
                bordermode = 'valid'
            else:
                bordermode = 'full'
        else:
            bordermode = dnnpad

        return dnnpaddable, bordermode, trim

    # Trims the edges of the convolution result to compensate for zero padding in full convolution.
    def _trim(self, inp=None, numconvs=2):
        # Say: h = x * W (full convolution). h.shape[i] = x.shape[i] + W.shape[i] - 1
        # Remember,
        #   inp.shape = (numimages, z, fmapsout, y, x)      [3D]
        #   inp.shape = (numimages, fmapsout, y, x)         [2D]
        # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
        # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]

        # Parse
        if not inp:
            inpgiven = False
            inp = self.xr
        else:
            inpgiven = True

        # 2D
        if self.dim == 2:
            trimy, trimx = [numconvs * (self.W.get_value().shape[i] - 1) / 2 for i in [2, 3]]  # trimmersize = [trimy, trimx]
            offsety, offsetx = [int(1 - (self.W.get_value().shape[i] % 2)) for i in [2, 3]]
            # For nx1 or 1xn convolutions, trimy or trimx is 0. Indexing 0:0 doesn't make sense in python, hence the
            # use of dictionaries
            out = inp[::, ::, (trimy):{-trimy - offsety: -trimy - offsety, 0: None}[-trimy - offsety],
                  (trimx):{-trimx - offsetx: -trimx - offsetx, 0: None}[-trimx - offsetx]]
        # 3D
        elif self.dim == 3:
            trimz, trimy, trimx = [numconvs * (self.W.get_value().shape[i] - 1) / 2 for i in [1, 3, 4]]
            offsetz, offsety, offsetx = [int(1 - (self.W.get_value().shape[i] % 2)) for i in [1, 3, 4]]
            out = inp[::, (trimz):{-trimz - offsetz: -trimz - offsetz, 0: None}[-trimz - offsetz], ::,
                  (trimy):{-trimy - offsety: -trimy - offsety, 0: None}[-trimy - offsety],
                  (trimx):{-trimx - offsetx: -trimx - offsetx, 0: None}[-trimx - offsetx]]
        # Raise
        else:
            raise NotImplementedError('Invalid network dimension.')

        # Assign return value (out) to self.xr only if no input given (i.e. inp = self.xr)
        if not inpgiven:
            self.xr = out
            return self.xr
        else:
            return out

    # Method to activate layer
    def activate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = True

        if what == 'dec' or what == 'all':
            self.decoderactive = True

    # Method to deactivate layer
    def deactivate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = False

        if what == 'dec' or what == 'all':
            self.decoderactive = False

    # Method to infer output shape
    def inferoutshape(self, inpshape=None, checkinput=True):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Check if input shape valid
        if checkinput:
            assert inpshape[(1 if self.inpdim == 4 else 2)] == self.fmapsin or \
                   inpshape[(1 if self.inpdim == 4 else 2)] is None, "Number of input channels must match that of " \
                                                                     "the layer."

        if self.inpdim == 4:
            # Compute y and x from input
            y, x = [((inpshape[sid] + (1 if self.convmode is 'full' else 0 if self.convmode is 'same' else -1) *
                    (self.kersize[kid] - 1) + (self.stride[kid] - 1) + sum(self.padding[kid])) // (self.stride[kid])
                     if inpshape[sid] is not None else None) for sid, kid in zip([2, 3], [0, 1])]
            # Fetch batchsize and number of output maps
            fmapsout = self.fmapsout
            batchsize = inpshape[0]

            return [batchsize, fmapsout, y, x]

        elif self.inpdim == 5:
            assert len(self.kersize) == 5 or self.issequence, "Layer must be 3D convolutional or sequential " \
                                                              "for 5D inputs."
            # Compute y, x and z from input
            y, x, z = [((inpshape[sid] + (1 if self.convmode is 'full' else 0 if self.convmode is 'same' else -1) *
                        (self.kersize[kid] - 1) + (self.stride[kid] - 1) + sum(self.padding[kid])) // (self.stride[kid])
                        if inpshape[sid] is not None else None) for sid, kid in zip([3, 4, 1], [0, 1, 2])]
            # Fetch batchsize and number of output maps
            fmapsout = self.fmapsout
            batchsize = inpshape[0]
            return [batchsize, z, fmapsout, y, x]

        pass

    # Extends the layer by making a new layer with filters of both self and other
    def __mul__(self, other):
        # Check type
        # Other can be an integer (for net2widernet) or another layer. Act accordingly.
        # FIXME: This behaviour (convlayer * convlayer = convlayer) is somewhat absurd. To make things worse,
        # FIXME: __pow__ behaves like the user would expect __mul__ to.
        if isinstance(other, convlayer):

            # Check if layers compatible:
            # Check if dimensions same
            if not (self.dim == other.dim):
                raise TypeError('Can not pair (multiply) layers of different dimensions.')

            # Check if kernel size same
            if not (self.kersize == other.kersize):
                raise TypeError('Kernel sizes do not match.')
            else:
                kersize = self.kersize

            # Check if fmapsin same
            if not (self.fmapsin == other.fmapsin):
                raise TypeError('Number of input maps (fmapsin) do not match.')
            else:
                fmapsin = self.fmapsin

            # Check if activations same
            if not (self.activation == other.activation):
                warn('Activation functions do not match. Using that of self.')
            activation = self.activation

            # Extend W
            # Reminder:
            # W.shape = (fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])     [3D]
            # W.shape = (fmapsout, fmapsin, kersize[0], kersize[1])                 [2D]
            W = T.concatenate([self.W, other.W], axis=0)
            Wc = T.concatenate([self.Wc, other.Wc], axis=0)
            Wgc = [max([self.Wgc[0], other.Wgc[0]]), min([self.Wgc[1], other.Wgc[1]])]
            fmapsout = self.fmapsout + other.fmapsout

            # Use zero bias only if both layers do
            zerobias = self.zerobias and other.zerobias

            if not zerobias:
                # Extend b
                b = T.concatenate([self.b, other.b], axis=0)
                bc = T.concatenate([self.bc, other.bc], axis=0)
            else:
                b = None
                bc = None

            bgc = [max([self.bgc[0], other.bgc[0]]), min([self.bgc[1], other.bgc[1]])]

            # Decoder flag: set true if decoder active for both of the two layers (this is required since bp is defined
            # only if decoders active)
            makedecoder = self.decoderactive and other.decoderactive

            # tied biases: set to false only if both layers have untied biases
            if (not self.tiedbiases) and (not other.tiedbiases) and makedecoder:
                tiedbiases = False
                # new decoder bias is the mean of the old two
                bp = T.mean(T.stacklists([self.bp, other.bp]), axis=0)
                # new decoder mask is the 'and' of the old two
                bpc = self.bpc * other.bpc
                # Gradient clips
                bpgc = [max([self.bpgc[0], other.bpgc[0]]), min([self.bpgc[1], other.bpgc[1]])]
            else:
                tiedbiases = True
                bp = None
                bpc = None
                bpgc = [max([self.bpgc[0], other.bpgc[0]]), min([self.bpgc[1], other.bpgc[1]])]
            return convlayer(fmapsin, fmapsout, kersize, activation=activation, makedecoder=makedecoder,
                             tiedbiases=tiedbiases, W=W, b=b, bp=bp, Wc=Wc, bc=bc, bpc=bpc, Wgc=Wgc, bgc=bgc, bpgc=bpgc)
        elif isinstance(other, int):
            assert other > 0, "Convlayer can only be multiplied by a positive integer."
            # Meta info (for temporary ref)
            fmapsin = self.fmapsin
            fmapsout = self.fmapsout + other
            kersize = self.kersize
            activation = self.activation
            makedecoder = self.decoderactive
            zerobias = self.zerobias
            tiedbiases = self.tiedbiases
            convmode = self.convmode
            Wgc = self.Wgc
            bgc = self.bgc
            bpgc = self.bpgc
            inpshape = self.inpshape
            allowsequences = self.allowsequences

            # Convolution Kernels
            # Fetch numerical value of the layer weights
            numW = self.W.get_value()
            catW = netools.idkernel(shape=[other] + list(numW.shape[1:]), shared=False)
            # Rescale and concatenate
            W = th.shared(
                np.concatenate(((self.fmapsout / float(fmapsout)) * numW, (other / float(fmapsout)) * catW), axis=0))
            numWc = self.Wc.get_value()
            catWc = np.ones_like(catW, dtype=th.config.floatX)
            Wc = th.shared(np.concatenate((numWc, catWc), axis=0))

            if not zerobias:
                # Biases (init new biases with zeros)
                numb = self.b.get_value()
                catb = np.zeros(shape=(other,), dtype=th.config.floatX)
                b = th.shared(np.concatenate((numb, catb), axis=0))
                # Bias connectivities
                numbc = self.bc.get_value()
                catbc = np.zeros(shape=(other,), dtype=th.config.floatX)
                bc = th.shared(np.concatenate((numbc, catbc), axis=0))
            else:
                b = bc = None

            # Decoder bias
            if makedecoder and not tiedbiases:
                # Decoder biases do not change because the number of input
                bp = self.bp
                bpc = self.bpc
            else:
                bp = None
                bpc = None

            return convlayer(fmapsin, fmapsout, kersize, activation=activation, makedecoder=makedecoder,
                             tiedbiases=tiedbiases, W=W, b=b, bp=bp, Wc=Wc, bc=bc, bpc=bpc, Wgc=Wgc, bgc=bgc, bpgc=bpgc,
                             zerobias=zerobias, convmode=convmode, inpshape=inpshape,
                             allowsequences=allowsequences)
        else:
            super(self, convlayer).__add__(other)

    # Define exponentiation (as parallel pairing of two layers)
    def __pow__(self, other, modulo=None):

        # Assertions and meta assignments
        assert isinstance(other, convlayer), "Layer type mismatch."

        # Kernel size must be same. This can be made more general by padding the smaller kernel with zeros.
        assert other.kersize == self.kersize, "Layers being paired must have identical kernel size and dimension."
        kersize = self.kersize

        # Activations must be the same
        assert self.activation == other.activation, "Activation functions must be the same."
        activation = self.activation

        # Generate zero blocks (dimension based)
        if self.dim == 2:
            zb1 = T.zeros(shape=(other.fmapsout, self.fmapsin, self.kersize[0], self.kersize[1]))
            zb2 = T.zeros(shape=(self.fmapsout, other.fmapsin, self.kersize[0], self.kersize[1]))
            inaxis = 1
        elif self.dim == 3:
            zb1 = T.zeros(shape=(other.fmapsout, self.kersize[2], self.fmapsin, self.kersize[0], self.kersize[1]))
            zb2 = T.zeros(shape=(self.fmapsout, self.kersize[2], other.fmapsin, self.kersize[0], self.kersize[1]))
            inaxis = 2
        else:
            raise NotImplementedError('Invalid network dimension.')

        # Extend W and Wc with zero blocks
        W = T.concatenate([T.concatenate([self.W, zb1], axis=0), T.concatenate([zb2, other.W], axis=0)], axis=inaxis)
        Wc = T.concatenate([T.concatenate([self.Wc, zb1], axis=0), T.concatenate([zb2, other.Wc], axis=0)], axis=inaxis)
        Wgc = [max([self.Wgc[0], other.Wgc[0]]), min([self.Wgc[1], other.Wgc[1]])]
        fmapsin = self.fmapsin + other.fmapsin
        fmapsout = self.fmapsout + other.fmapsout

        # Extend b and bc
        b = T.concatenate([self.b, other.b])
        bgc = [max([self.bgc[0], other.bgc[0]]), min([self.bgc[1], other.bgc[1]])]
        bc = T.concatenate([self.bc, other.bc])

        # Switch if decoder is active
        decoderactive = self.decoderactive and other.decoderactive

        # Extend untied decoder biases if decoders active
        if decoderactive and (not self.tiedbiases) and (not other.tiedbiases):
            tiedbiases = False
            bp = T.concatenate([self.bp, other.bp], axis=0)
            bpc = T.concatenate([self.bpc, other.bpc], axis=0)
            bpgc = [max([self.bpgc[0], other.bpgc[0]]), min([self.bpgc[1], other.bpgc[1]])]
        else:
            tiedbiases = True
            bp = None
            bpc = None
            bpgc = [max([self.bpgc[0], other.bpgc[0]]), min([self.bpgc[1], other.bpgc[1]])]

        return convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize, activation=activation,
                         makedecoder=decoderactive, tiedbiases=tiedbiases, W=W, b=b, bp=bp, Wc=Wc, bc=bc, bpc=bpc,
                         Wgc=Wgc, bgc=bgc, bpgc=bpgc)

    # Define division. Multiplication adds output maps, ergo division adds input maps
    def __div__(self, other):
        if isinstance(other, int):
            # Compute the new number of input feature maps
            fmapsin = self.fmapsin + other
            fmapsout = self.fmapsout
            kersize = self.kersize
            activation = self.activation
            makedecoder = self.decoderactive
            tiedbiases = self.tiedbiases
            zerobias = self.zerobias
            convmode = self.convmode
            Wgc = self.Wgc
            bgc = self.bgc
            bpgc = self.bpgc
            allowsequences = self.allowsequences

            # Fetch numerical weights
            # Fetch numerical value of the layer weights
            numW = self.W.get_value()
            # Make a kernel to concatenate with
            catW = np.tile(numW.mean(axis=(1 if self.inpdim == 4 else 2), keepdims=True),
                           reps=([1, other, 1, 1] if self.inpdim == 4 else [1, 1, other, 1, 1]))
            # Compute the final kernel
            W = th.shared(np.concatenate(((self.fmapsin / float(fmapsin)) * numW, (other / float(fmapsin)) * catW),
                                         axis=(1 if self.inpdim == 4 else 2)))
            # Conv. kernel connectivities (default new values to 1)
            numWc = self.Wc.get_value()
            catWc = np.ones_like(catW, dtype=th.config.floatX)
            Wc = th.shared(np.concatenate((numWc, catWc), axis=(1 if self.inpdim == 4 else 2)))

            # Leave bias unchanged
            b = self.b
            bc = self.bc

            # Get decoder biases
            if self.decoderactive and not self.tiedbiases:
                numbp = self.bp.get_value()
                catbp = np.zeros(shape=(other,), dtype=th.config.floatX)
                bp = th.shared(np.concatenate((numbp, catbp), axis=0))

                numbpc = self.bpc.get_value()
                catbpc = np.ones_like(catbp, dtype=th.config.floatX)
                bpc = th.shared(np.concatenate((numbpc, catbpc), axis=0))
            else:
                bp = None
                bpc = None

            return convlayer(fmapsin, fmapsout, kersize, activation=activation, makedecoder=makedecoder,
                             tiedbiases=tiedbiases, W=W, b=b, bp=bp, Wc=Wc, bc=bc, bpc=bpc, Wgc=Wgc, bgc=bgc, bpgc=bpgc,
                             zerobias=zerobias, convmode=convmode,
                             allowsequences=allowsequences)

        else:
            raise NotImplementedError("Convlayer can only be divided by an integer")
        pass


class poollayer(layer):
    """ General Max-pooling Layer """

    # Constructor
    def __init__(self, ds, stride=None, ignoreborder=True, padding=(0, 0), switchmode=None, switch=None,
                 allowsequences=True, makedecoder=True, inpshape=None):
        """
        :type ds: 2- or 3-tuple for 2 or 3 dimensional network
        :param ds: tuple of downsampling ratios

        :type stride: tuple or list
        :param stride: Pooling strides. Defaults to ds (i.e. non-overlapping regions).

        :type padding: tuple or list
        :param padding: Input padding. Handled by Theano's max_pool_2D.

        :type ignoreborder: bool
        :param ignoreborder: Whether to ignore borders while pooling. Equivalent to ignore_border in Theano's
                             max_pool_2d.

        :type switchmode: str
        :param switchmode: Whether to use switches for unpooling. Layer must be 2 dimensional to use switches and the
                       possible options are:
                         None   :   No switches used
                        'hard'  :   Hard switches (1 for the max value, 0 for everything else in the window)
                        'soft'  :   Soft switches (proportion preserving switches: 1 for max, 0.5 for a value half that
                                    of max, etc.)

        :type switch: theano.tensor.var.TensorVariable
        :param switch: Optional input slot for a switch variable. Overrides switch variable computed in layer.

        :type allowsequences: bool
        :param allowsequences: Whether to allow sequences. If set to true and ds[2] = 1, 3D spatiotemporal data is
                               spatially pooled.

        :type inpshape: list or tuple
        :param inpshape: Expected input/output shape.

        """

        # Initialize super class
        super(poollayer, self).__init__()

        # Input check
        assert len(ds) == 2 or len(ds) == 3, "ds can only be a vector/list of length 2 or 3."

        # Meta
        self.ds = ds
        self.stride = self.ds if stride is None else stride
        self.ignoreborder = ignoreborder
        self.padding = padding
        self.decoderactive = makedecoder
        self.encoderactive = True
        self.switchmode = switchmode
        self.allowsequences = allowsequences

        # Define dummy parameter lists for compatibility with layertrain
        self.params = []
        self.cparams = []

        # Parse network dimension (and whether the input a sequence)
        self.issequence = len(self.ds) == 3 and self.ds[2] == 1 and self.allowsequences
        self.dim = (2 if len(self.ds) == 2 or (self.issequence and self.allowsequences) else 3)

        if self.dim == 2:
            self.inpdim = (4 if not self.issequence else 5)
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimension: {}. Supported: 2D or 3D.'.format(self.dim))

        # Switches
        # Check if switched unpooling possible
        assert not (switchmode is not None and self.dim == 3), "Switched unpooling implemented in 2D only."
        assert switchmode in [None, 'soft', 'hard'], "Implemented switch modes are 'soft' and 'hard'."

        # Check if switch variable provided
        if switch is not None:
            self.switch = switch
            self.switchmode = 'given'
        else:
            self.switch = T.tensor('floatX', [False, False, False, False], name='sw:' + str(id(self)))

        # Input and output shapes
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inpshape must match equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Containers (see feedforward() for input shapes)
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))
        # Reconstructed input
        self.xr = T.tensor('floatX', [False, ] * self.inpdim, name='xr:' + str(id(self)))

        self.layerinfo = "[maxpool by {} kernel]".format(self.ds)

    # Feed forward through the layer (pool)
    def feedforward(self, inp=None, reshape=False):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        __old__ = True

        if not __old__:
            self.y = A.pool(inp=inp, ds=self.ds, stride=self.stride, padding=self.padding, dim=self.dim,
                            ignoreborder=self.ignoreborder, issequence=self.issequence)
            return self.y

        if __old__:
            # Reshape sequence data
            # Log initial shape
            inpshape = inp.shape
            if self.issequence:
                # This makes inp.shape = (batchnum * T, fmapsin, y, x) out of (batchnum, T, fmapsin, y, x)
                inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)

            # Reshape if requested
            if self.dim == 3 and reshape:
                inp = inp.dimshuffle(0, 2, 3, 4, 1)

            # Get poolin'
            if self.dim == 2:
                # Make switches (or skip if a switch variable was given
                if self.switchmode is 'given':
                    # Nothing to do
                    pass

                elif self.switchmode is None:
                    # Trivial switch of ones
                    self.switch = T.ones_like(inp)

                elif self.switchmode is 'hard':
                    # Convert input to neighborhoods
                    inpnebs = nebs.images2neibs(inp, self.ds, neib_step=self.ds, mode='valid')
                    # Find switch corresponding to the max value in rows of inpnebs and set to 1. All other switches should
                    # be set to 0
                    sw = T.set_subtensor(
                        T.zeros_like(inpnebs)[T.arange(0, inpnebs.shape[0]), T.argmax(inpnebs, axis=1)], 1.)
                    # Reshape sw from neighborhoods to image and assign to switch
                    self.switch = nebs.neibs2images(sw, self.ds, inp.shape, mode='valid')

                elif self.switchmode is 'soft':
                    # Convert input to neighborhoods
                    inpnebs = nebs.images2neibs(inp, self.ds, neib_step=self.ds, mode='valid')
                    # Set soft switches by normalizing inpnebs row-wise by the maximum
                    sw = inpnebs / T.max(inpnebs, axis=1, keepdims=True)
                    # Reshape from neighborhoods and assign to switch
                    self.switch = nebs.neibs2images(sw, self.ds, inp.shape, mode='valid')

                # Compute downsampling ratio (based on whether the input is sequential)
                ds = self.ds[0:2]
                st = self.stride[0:2]
                pad = self.padding[0:2]

                # Downsample
                y = downsample.max_pool_2d(input=inp, ds=ds, st=st, padding=pad, ignore_border=self.ignoreborder)

                # Reshape sequence data
                if self.issequence:
                    # Compute symbolic pool output length
                    if self.ignoreborder:
                        pooleny, poolenx = \
                            [T.floor((inpshape[tensorindex] + 2 * pad[index] - ds[index] + st[index])/st[index])
                             for index, tensorindex in enumerate([3, 4])]

                    else:
                        poolen = [None, None]

                        for index, tensorindex in enumerate([3, 4]):
                            if st[index] >= ds[index]:
                                poolen[index] = T.floor((inpshape[tensorindex] + st[index] - 1)/st[index])
                            else:
                                plen = T.floor((inpshape[tensorindex] - ds[index] + st[index] - 1)/st[index])
                                poolen[index] = T.switch(plen > 0, plen, 0)

                        pooleny, poolenx = poolen

                    self.y = y.reshape((inpshape[0], inpshape[1], inpshape[2], pooleny, poolenx), ndim=5)

                else:
                    self.y = y

                return self.y

            elif self.dim == 3:
                # Theano lesson: downsample.max_pool_2d downsamples the last 2 dimensions in a tensor. To pool in 3D, the z
                # dimension needs to be pooled separately after 'rotating' the tensor appropriately such that the z axis is
                # the last dimension.

                # parse downsampling ratio, stride and padding
                dsyx = self.ds[0:2]
                styx = self.stride[0:2]
                padyx = self.padding[0:2]

                ds0z = (1, self.ds[2])
                st0z = (1, self.stride[2])
                pad0z = (0, self.padding[2])

                # Dowsnample yx
                H = downsample.max_pool_2d(input=inp, ds=dsyx, st=styx, padding=padyx)
                # Rotate tensor
                H = H.dimshuffle(0, 2, 3, 4, 1)
                # Downsample 0z
                H = downsample.max_pool_2d(input=H, ds=ds0z, st=st0z, padding=pad0z)
                # Undo rotate tensor
                self.y = H.dimshuffle(0, 4, 1, 2, 3)
                return self.y

    # Feed forward through the decoder layer (unpool, relevant for convolutional autoencoders)
    def decoderfeedforward(self, inp=None, reshape=False):
        # Argument inp is expected of the form:
        #    inp.shape = (numimages, z, fmapsin, y, x)     [3D]
        #    inp.shape = (numimages, fmapsin, y, x)        [2D]
        # Setting reshape to True assumes that the 3D input is of the form (numimages, fmapsin, y, x, z)

        # Parse input
        if inp is None:
            inp = self.y
        else:
            self.y = inp

        # Return input if decoder not active
        if not self.decoderactive:
            self.xr = inp
            return inp

        assert self.stride == self.ds, "Strided upsampling not supported."

        # Reshape sequence data
        # Log initial shape
        inpshape = inp.shape
        if self.issequence:
            # This makes inp.shape = (batchnum * T, fmapsin, y, x) out of (batchnum, T, fmapsin, y, x)
            inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)

        # Reshape if requested
        if self.dim == 3 and reshape:
            inp = inp.dimshuffle(0, 2, 3, 4, 1)

        if self.issequence:
            ds = self.ds[0:2]
        else:
            ds = self.ds

        # Get unpoolin'
        if self.dim == 2:
            # Upsample
            xr = self.switch * T.repeat(T.repeat(inp, ds[0], axis=2), ds[1], axis=3)

            # Reshape sequence data
            if self.issequence:
                self.xr = xr.reshape((inpshape[0], inpshape[1], inpshape[2], inpshape[3]*ds[0], inpshape[4]*ds[1]),
                                     ndim=5)
            else:
                self.xr = xr

            return self.xr

        elif self.dim == 3:
            # parse upsampling ratio
            usrat = (1, ds[2], 1, ds[0:2])

            # Upsample
            self.xr = T.repeat(T.repeat(T.repeat(inp, ds[0], axis=3), ds[1], axis=4), ds[2], axis=1)

            return self.xr

    # Method to activate layer
    def activate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = True

        if what == 'dec' or what == 'all':
            self.decoderactive = True

    # Method to deactivate layer
    def deactivate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = False

        if what == 'dec' or what == 'all':
            self.decoderactive = False

    # Infer output shape
    def inferoutshape(self, inpshape=None, checkinput=False):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Process
        if self.inpdim == 4:
            if self.ignoreborder:
                y, x = [int(np.floor((inpshape[sid] + 2 * self.padding[kid] - self.ds[kid] + self.stride[kid]) /
                                     self.stride[kid])) if inpshape[sid] is not None else None
                        for sid, kid in zip([2, 3], [0, 1])]
            else:
                plen = [None, None]
                for sid, kid in zip([2, 3], [0, 1]):
                    if self.stride[kid] >= self.ds[kid]:
                        plen[kid] = int(np.floor((inpshape[sid] + self.stride[kid] - 1) / self.stride[kid])) \
                            if inpshape[sid] is not None else None
                    else:
                        plen[kid] = np.maximum(0, np.floor((inpshape[sid] - self.ds[kid] + self.stride[kid] - 1) /
                                                           self.stride[kid])) if inpshape[sid] is not None else None
                y, x = plen

            fmapsout = inpshape[1]
            batchsize = inpshape[0]

            return [batchsize, fmapsout, y, x]

        elif self.inpdim == 5:
            if self.ignoreborder:
                y, x, z = [int(np.floor((inpshape[sid] + 2 * self.padding[kid] - self.ds[kid] + self.stride[kid]) /
                                        self.stride[kid])) if inpshape[sid] is not None else None
                           for sid, kid in zip([3, 4, 1], [0, 1, 2])]
            else:
                plen = [None, None, None]
                for sid, kid in zip([3, 4, 1], [0, 1, 2]):
                    if self.stride[kid] >= self.ds[kid]:
                        plen[kid] = int(np.floor((inpshape[sid] + self.stride[kid] - 1) / self.stride[kid])) \
                            if inpshape[sid] is not None else None
                    else:
                        plen[kid] = np.maximum(0, np.floor((inpshape[sid] - self.ds[kid] + self.stride[kid] - 1) /
                                                           self.stride[kid])) if inpshape[sid] is not None else None
                y, x, z = plen

            fmapsout = inpshape[2]
            batchsize = inpshape[0]

            return [batchsize, z, fmapsout, y, x]


class upsamplelayer(layer):
    """Unpool/upsample layer with or without interpolation"""

    def __init__(self, us, interpolate=False, allowsequences=True, fmapsin=None, activation=netools.linear(),
                 inpshape=None):
        """
        :type us: list or tuple
        :param us: Upsampling ratio.

        :type interpolate: bool
        :param interpolate: Whether to interpolate (i.e. convolve with a normalizede unit convolution kernel)

        :type allowsequences: bool
        :param allowsequences: Whether input can allowed to be a sequence (i.e. apply the upsampling framewise).
                               us must be [n, m, 1] where n and m are positive integers.

        :type fmapsin: int
        :param fmapsin: Number of input feature maps. Required for interpolation, but can also be infered from the
                        input shape.

        :type inpshape: list or tuple
        :param inpshape: Input shape
        :return:
        """

        # Construct superclass
        super(upsamplelayer, self).__init__()

        # Meta
        self.us = list(us)
        self.interpolate = interpolate
        self.allowsequences = allowsequences
        self.fmapsin = fmapsin
        self.activation = activation

        # Determine data and input dimensions
        self.inpdim = {2: 4, 3: 5}[len(us)]
        self.issequence = self.allowsequences and self.us[-1] == 1 if self.inpdim == 5 else False
        self.dim = 2 if (self.inpdim == 4 or self.issequence) else 3

        # Shape inference
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            self.inpshape = inpshape

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name="x:" + str(id(self)))
        self.y = T.tensor('floatX', [False, ] * self.outdim, name="y:" + str(id(self)))

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        if checkinput:
            assert len(inpshape) == self.inpdim, "Length of the provided input shape does not match the " \
                                                 "number of input dimensions."

        if self.dim == 2:
            outshape = cp.copy(inpshape)
            outshape[-2:] = [shp * us if shp is not None else None for us, shp in zip(self.us[0:2], outshape[-2:])]
        elif self.dim == 3:
            outshape = cp.copy(inpshape)
            outshape[1] = outshape[1] * self.us[2] if outshape[1] is not None else None
            outshape[-2:] = [shp * us if shp is not None else None for us, shp in zip(self.us[0:2], outshape[-2:])]
        else:
            raise NotImplementedError

        return outshape

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        if not self.encoderactive:
            self.y = inp
            return inp

        __old__ = True

        if not __old__:
            usd = A.unpool(inp, us=self.us, dim=self.dim, issequence=self.issequence)

            # Interpolate if required
            if self.interpolate:
                # Make convolution kernel for interpolation.
                if self.dim == 2:
                    # Fetch number of feature maps
                    self.fmapsin = self.inpshape[(1 if self.inpdim == 4 else 2)] if self.fmapsin is None else self.fmapsin

                    assert self.fmapsin is not None, "Number of input feature maps could not be inferred."

                    # Make conv-kernels
                    numconvker = netutils.setkernel(inds=[[fm, fm] for fm in range(self.fmapsin)],
                                                    subkernel=(1./(self.us[0] * self.us[1])),
                                                    out=np.zeros(
                                                        shape=(self.fmapsin, self.fmapsin, self.us[0], self.us[1]))).\
                        astype(th.config.floatX)
                    convker = th.shared(value=numconvker)

                elif self.dim == 3:
                    # Make convolution kernel for interpolation.
                    # Fetch number of feature maps
                    self.fmapsin = self.inpshape[(1 if self.inpdim == 4 else 2)] if self.fmapsin is None else self.fmapsin

                    assert self.fmapsin is not None, "Number of input feature maps could not be inferred."

                    # Make conv-kernels
                    numconvker = netutils.setkernel(inds=[[fm, fm] for fm in range(self.fmapsin)],
                                                    subkernel=(1./(self.us[0] * self.us[1] * self.us[2])),
                                                    out=np.zeros(shape=(self.fmapsin, self.us[2], self.fmapsin,
                                                                        self.us[0], self.us[1]))).\
                        astype(th.config.floatX)
                    convker = th.shared(value=numconvker)

                else:
                    raise NotImplementedError

                # Convolve to interpolate
                usd = A.conv(usd, convker, convmode='same')

            # Apply activation function
            usd = self.activation(usd)
            self.y = usd
            return self.y

        if __old__:
            # Reshape sequential input
            # Check if input really is sequential
            reallyissequence = self.issequence and inp.ndim == 5
            # Log initial shape
            inpshape = inp.shape
            if self.issequence:
                if reallyissequence:
                    # This makes inp.shape = (batchnum * T, fmapsin, y, x) out of (batchnum, T, fmapsin, y, x)
                    inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)
                else:
                    warn("upsamplelayer expected a 5D sequential input, but got 4D non-sequential instead.")

            # Trim us if required
            us = self.us[0:2] if self.issequence else self.us

            # Unpool (similar to pool layer decoder)
            if self.dim == 2:
                # Upsample
                usd = T.repeat(T.repeat(inp, us[0], axis=2), us[1], axis=3)

                if self.interpolate:
                    # Make convolution kernel for interpolation.
                    # Fetch number of feature maps
                    self.fmapsin = self.inpshape[(1 if self.inpdim == 4 else 2)] if self.fmapsin is None else self.fmapsin
                    assert self.fmapsin is not None, "Number of input feature maps could not be inferred."
                    # Make conv-kernels
                    numconvker = netutils.setkernel(inds=[[fm, fm] for fm in range(self.fmapsin)],
                                                    subkernel=(1./(us[0] * us[1])),
                                                    out=np.zeros(shape=(self.fmapsin, self.fmapsin, us[0], us[1]))).\
                        astype(th.config.floatX)
                    convker = th.shared(value=numconvker)
                    # Convolve
                    usd = conv.conv2d(input=usd, filters=convker, border_mode='full',
                                      filter_shape=(self.fmapsin, self.fmapsin, us[0], us[1]))
                    # Trim edges
                    trimy, trimx = [(numconvker.shape[i] - 1) / 2 for i in [2, 3]]  # trimmersize = [trimy, trimx]
                    offsety, offsetx = [1 if dimus % 2 == 0 else 0 for dimus in us]
                    usd = usd[::, ::, (trimy):{-trimy - offsety: -trimy - offsety, 0: None}[-trimy - offsety],
                      (trimx):{-trimx - offsetx: -trimx - offsetx, 0: None}[-trimx - offsetx]]

                # Activation
                usd = self.activation(usd)

                # Reshape sequence data
                if reallyissequence:
                    y = usd.reshape((inpshape[0], inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=5)
                else:
                    y = usd

                self.y = y
                return self.y

            elif self.dim == 3:
                usd = T.repeat(T.repeat(T.repeat(inp, us[0], axis=3), us[1], axis=4), us[2], axis=1)

                if self.interpolate:
                    # Make convolution kernel for interpolation.
                    # Fetch number of feature maps
                    self.fmapsin = self.inpshape[(1 if self.inpdim == 4 else 2)] if self.fmapsin is None else self.fmapsin
                    assert self.fmapsin is not None, "Number of input feature maps could not be inferred."
                    # Make conv-kernels
                    numconvker = netutils.setkernel(inds=[[fm, fm] for fm in range(self.fmapsin)],
                                                    subkernel=(1./(us[0] * us[1] * us[2])),
                                                    out=np.zeros(shape=(self.fmapsin, us[2], self.fmapsin, us[0], us[1]))).\
                        astype(th.config.floatX)
                    convker = th.shared(value=numconvker)
                    # Convolve
                    usd = conv3d2d.conv3d(signals=usd, filters=convker, border_mode='full',
                                          filters_shape=(self.fmapsin, us[2], self.fmapsin, us[0], us[1]))
                    # Trim edges
                    trimz, trimy, trimx = [(numconvker.shape[i] - 1) / 2 for i in [1, 3, 4]]
                    offsety, offsetx, offsetz = [1 if dimus % 2 == 0 else 0 for dimus in us]
                    usd = usd[::, (trimz):{-trimz - offsetz: -trimz - offsetz, 0: None}[-trimz - offsetz], ::,
                      (trimy):{-trimy - offsety: -trimy - offsety, 0: None}[-trimy - offsety],
                      (trimx):{-trimx - offsetx: -trimx - offsetx, 0: None}[-trimx - offsetx]]

                usd = self.activation(usd)

                self.y = usd
                return self.y

            else:
                raise NotImplementedError("Data must be 2 or 3D.")


class softmax(layer):
    """ Framework embedded softmax function without learnable parameters """

    def __init__(self, dim, onehot=False, inpshape=None):
        """
        :type dim: int
        :param dim: Layer dimensionality. 1 for vectors, 2 for images and 3 for volumetric data

        :type onehot: bool
        :param onehot: Whether to encode one-hot for prediction

        :type inpshape: tuple or list
        :param inpshape: Shape of the expected input

        """

        # Initialize super class
        super(softmax, self).__init__()

        # Meta
        self.decoderactive = False
        self.encoderactive = True
        self.dim = dim
        self.onehot = onehot

        if self.dim == 1:
            self.inpdim = 2
        elif self.dim == 2:
            self.inpdim = 4
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimensionality: {}. Supported: 1D, 2D and 3D.'.format(self.dim))

        # Dummy parameters for compatibility with layertrain
        self.params = []
        self.cparams = []

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Containers
        if self.dim == 1:
            # Input
            self.x = T.matrix('x:' + str(id(self)))
            # Output
            self.y = T.matrix('y:' + str(id(self)))
        elif self.dim == 2 or self.dim == 3:
            # Input
            self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
            # Output
            self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))
        else:
            raise NotImplementedError

    # Feedforward
    def feedforward(self, inp=None, predict=None):
        # inp is expected of the form
        #   inp.shape = (numimages, sigsin)

        # Parse Input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        if predict is None:
            predict = self.onehot

        __old__ = True

        if not __old__:
            self.y = A.softmax(inp, dim=self.dim, predict=predict, issequence=self.issequence)
            return self.y

        if __old__:
            # FFD for dimensions 1 and 2
            if self.dim == 1 or self.dim == 2:
                # Using the numerically stable implementation (along the channel axis):
                ex = T.exp(inp - T.max(inp, axis=1, keepdims=True))
                self.y = ex / T.sum(ex, axis=1, keepdims=True)

                # One hot encoding for prediction
                if predict:
                    self.y = T.argmax(self.y, axis=1)
                # Return
                return self.y

            elif self.dim == 3:
                # Stable implementation again, this time along axis = 2 (channel axis)
                ex = T.exp(inp - T.max(inp, axis=2, keepdims=True))
                self.y = ex / T.sum(ex, axis=2, keepdims=True)

                # One hot encoding for prediction
                if predict:
                    self.y = T.argmax(self.y, axis=2)
                # Return
                return self.y

    # Decoder feedforward. (do not use / not functional)
    def decoderfeedforward(self, inp=None):

        # Parse input (so that the decoder returns self.y and not None when no input is provided)
        if inp is None:
            inp = self.y
        else:
            self.y = inp

        # Warn
        warn('Decoder not implemented for softmax.')

        # Return
        return inp

    # Method to activate layer
    def activate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = True

        if what == 'dec' or what == 'all':
            self.decoderactive = True

    # Method to deactivate layer
    def deactivate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = False

        if what == 'dec' or what == 'all':
            self.decoderactive = False

    # Infer output shape
    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Set number of channels to 1 if onehot
        if self.onehot:
            inpshape[1] = 1

        return inpshape


class noiselayer(layer):
    """ General Noising Layer """

    # Constructor
    def __init__(self, noisetype=None, sigma=None, n=None, p=None, dim=2, thicken=True, makedecoder=False, rngseed=42,
                 inpshape=None):
        """
        :type noisetype: str
        :param noisetype: Possible keys: 'normal', 'binomial'.

        :type sigma: float
        :param sigma: std for normal noise

        :type n: float
        :param n: n for binomial (salt and pepper) noise

        :type p: float
        :param p: p for binomial (salt and pepper) noise (also the dropout amount)

        :type dim: int
        :param dim: Dimensionality of the layer

        :type thicken: bool
        :param thicken: (in Hinton speak) whether to divide the activations with the dropout amount (p)

        :type makedecoder: bool
        :param makedecoder: Noises in the decoder layer when set to True

        :type inpshape: tuple or list
        :param inpshape: Shape of the expected input
        """

        # Initialize super class
        super(noiselayer, self).__init__()

        # Meta
        self.thicken = thicken
        self.decoderactive = makedecoder
        self.encoderactive = True
        self.srng = RandomStreams(rngseed)

        # Dummy parameter list for compatibility with layertrain
        self.params = []
        self.cparams = []

        # Parse dimensionality
        self.dim = dim
        if self.dim == 1:
            self.inpdim = 2
        elif self.dim == 2:
            self.inpdim = 4
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError('Invalid layer dimensionality: {}. Supported: 1D, 2D and 3D.'.format(self.dim))

        if not noisetype:
            self.noisetype = 'normal'
        else:
            self.noisetype = noisetype

        if not sigma:
            self.sigma = 0.2
        else:
            self.sigma = sigma

        if not n:
            self.n = 1
        else:
            self.n = n

        if not p:
            self.p = 0.5
        else:
            self.p = p

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        self.outshape = self.inferoutshape()

        # Container for input (see feedforward() for input shapes) and output
        if self.dim == 2:
            # Input
            self.x = T.tensor('floatX', [False, False, False, False], name='x:' + str(id(self)))
            # Output
            self.y = T.tensor('floatX', [False, False, False, False], name='y:' + str(id(self)))
            # Reconstructed input
            self.xr = T.tensor('floatX', [False, False, False, False], name='xr:' + str(id(self)))

        elif self.dim == 3:
            # Input
            self.x = T.tensor('floatX', [False, False, False, False, False], name='x:' + str(id(self)))
            # Output
            self.y = T.tensor('floatX', [False, False, False, False, False], name='y:' + str(id(self)))
            # Reconstructed input
            self.xr = T.tensor('floatX', [False, False, False, False, False], name='xr:' + str(id(self)))

        elif self.dim == 1:
            # Input
            self.x = T.matrix('x:' + str(id(self)))
            # Output
            self.y = T.matrix('y:' + str(id(self)))
            # Reconstructed input
            self.xr = T.matrix('xr:' + str(id(self)))

    # Feedforward
    def feedforward(self, inp=None):
        # Parse Input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Check if encoder active
        if not self.encoderactive:
            self.y = inp
            return inp

        __old__ = True

        if not __old__:
            self.y = A.noise(inp, noisetype=self.noisetype, p=self.p, n=self.n, sigma=self.sigma, srng=self.srng)
            return self.y

        if __old__:
            # Noise Input
            out = self.noise(inp)

            # Thicken if necessary
            self.y = out / getattr(np, th.config.floatX)(self.p) if self.noisetype is 'binomial' and self.thicken else out

            # return
            return self.y

    # Decoder Feedforward
    def decoderfeedforward(self, inp=None):

        if inp is None:
            inp = self.y
        else:
            self.y = inp

        # Return input if decoder not active (do not assign to xr; it's not initialized if decoder inactive)
        if not self.decoderactive:
            self.xr = inp
            return inp

        # Decoder active, noise and return
        # Noise Input
        out = self.noise(inp)

        # Thicken if necessary
        self.xr = out / getattr(np, th.config.floatX)(self.p) if self.noisetype is 'binomial' and self.thicken else out

        # return
        return self.xr

    # Noise input
    def noise(self, inp):

        # Cast noising mask to floatX, because int * float32 = float64 which pulls things off the GPU
        if self.noisetype == 'normal':
            noisekernel = T.cast(self.srng.normal(size=inp.shape, std=self.sigma), dtype='floatX')
            return noisekernel + inp  # Additive gaussian noise
        elif self.noisetype == 'binomial':
            noisekernel = T.cast(self.srng.binomial(size=inp.shape, n=self.n, p=self.p), dtype='floatX')
            return noisekernel * inp  # Multiplicative binomial noise
        else:
            warn('noisetype invalid, using Gaussian noise with default sigma (0.2)')
            noisekernel = T.cast(self.srng.normal(size=inp.shape, std=self.sigma), dtype='floatX')
            return noisekernel + inp

    # Method to activate layer
    def activate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = True

        if what == 'dec' or what == 'all':
            self.decoderactive = True

    # Method to deactivate layer
    def deactivate(self, what='all'):
        if what == 'enc' or what == 'all':
            self.encoderactive = False

        if what == 'dec' or what == 'all':
            self.decoderactive = False