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
        self.outdim = pyk.delist([len(oshp) for oshp in pyk.list2listoflists(self.outshape)])
        self.numout = pyk.smartlen(self.outdim)

        # Set ghost parameter shapes (if there are any to begin with)
        # FIXME: One may have multiple getghostparamshape's in a layer
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

    # Apply states
    def applystates(self, states=None):
        """This method applies numerical (or theano shared) states to the layer."""
        if states is not None:
            # Convert to numeric (in case params is symbolic)
            states = netutils.sym2num(states)
            # Loop over all params, and set values
            for state, value in zip(self.state, states):
                state.set_value(value)

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
    def __init__(self, fmapsin, fmapsout, kersize, stride=None, padding=None, dilation=None, activation=netools.linear(),
                 alpha=None, makedecoder=False, zerobias=False, tiedbiases=True, convmode='same', allowsequences=True,
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

        :type dilation: tuple or list
        :param dilation: Dilation for dilated convolutions.

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

        # Debug Paramsh
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
                stride = [stride, ] * (self.dim + (0 if not self.issequence else 1)) if isinstance(stride, int) \
                    else stride
                assert len(stride) == len(self.kersize), "Stride and kersize must have the same length."
                self.stride = list(stride)
            else:
                warn("Convolution strides are presently not supported for 3D convolutions.")
                self.stride = [1, ] * (self.dim + (0 if not self.issequence else 1))

        # Parse dilation
        if dilation is None:
            self.dilation = [1, ] * (self.dim + (0 if not self.issequence else 1))
        else:
            if self.dim == 2:
                dilation = [dilation, ] * (self.dim + (0 if not self.issequence else 1)) if isinstance(dilation, int) \
                    else dilation
                assert len(dilation) == len(self.kersize), "Dilation and kersize must have the same length."
                assert self.stride == [1, 1], "Stride must be [1, 1] for dilated convolutions."
                self.dilation = list(dilation)
            else:
                warn("Dilated convolutions are presently not supported for 3D convolutions.")
                self.dilation = [1, ] * (self.dim + (0 if not self.issequence else 1))

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
                self.W = th.shared(
                    value=self.alpha * initscheme(shape=(fmapsout, kersize[2], fmapsin, kersize[0], kersize[1])),
                    name='convW:' + str(id(self)))
            else:
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
        self.Wc = netutils.getshared(value=(1. if Wc is None else Wc), like=self.W, name='convWc:' + str(id(self)))

        # Gradient clips
        if Wgc is None:
            self.Wgc = [-np.inf, np.inf]
        else:
            assert isinstance(Wgc, (list, np.ndarray)) and len(Wgc) == 2, "Weight filter gradient clips (Wgc) must " \
                                                                          "be a list with two elements."
            self.Wgc = Wgc

        # Biases and conn. mask
        self.b = netutils.getshared(value=(0. if b is None else b), shape=(fmapsout,), name='convb:' + str(id(self)))

        # Conn. mask
        if bc is None and not self.zerobias:
            self.bc = netutils.getshared(value=1., like=self.b, name='convbc:' + str(id(self)))
        elif self.zerobias and b is None:
            self.bc = netutils.getshared(value=0., like=self.b, name='convbc:' + str(id(self)))
        else:
            self.bc = netutils.getshared(value=bc, like=self.b, name='convbc:' + str(id(self)))

        # Gradient clips
        if not bgc and not self.zerobias:
            self.bgc = [-np.inf, np.inf]
        elif self.zerobias and bgc is None:
            self.bgc = [0, 0]
        else:
            assert isinstance(bgc, (list, np.ndarray)) and len(bgc) == 2, "Bias gradient clips (bgc) must " \
                                                                          "be a list with two elements."
            self.bgc = bgc

        # Fold Parameters
        self._params = [self.W] + ([self.b] if not self.zerobias else []) + self.extratrainables

        self._cparams = [self.Wc] + ([self.bc] if not self.zerobias else []) + self.extractrainables

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

        self.layerinfo = "{}-in >> {}-out w/ {} kernel".format(fmapsin, fmapsout, kersize)

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

        # Check if gradient masking is required
        filtergradmask = self.Wc if self.allowgradmask else None
        biasgradmask = self.bc if self.allowgradmask else None

        # Get PreActivation
        PA = A.conv(inp, self.W, stride=self.stride, dilation=self.dilation, padding=self.padding, bias=self.b,
                    filtergradmask=filtergradmask, biasgradmask=biasgradmask, filtergradclips=self.Wgc,
                    biasgradclips=self.bgc, dim=self.dim, convmode=self.convmode, issequence=self.issequence)

        # Apply activation function
        self.y = activation(PA)
        # Return
        return self.y

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


class poollayer(layer):
    """ General Max-pooling Layer """

    # Constructor
    def __init__(self, ds, stride=None, ignoreborder=True, padding=(0, 0), poolmode='max', switchmode=None,
                 switch=None, allowsequences=True, makedecoder=True, inpshape=None):
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
        assert len(ds) == 2 or len(ds) == 3 or ds == 'global', "ds can only be a vector/list of length 2 or 3 or " \
                                                               "'global'."

        # Check if global pooling
        if ds == 'global':
            # Only 2D global pooling is supported right now
            ds = ['global', 'global']
            # TODO continue

        # Meta
        self.ds = ds
        self.poolmode = poolmode
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

        if padding == 'auto':
            # Decide padding based on stride
            if self.stride == [2, 2]:
                if all([dsp % 2 == 1 for dsp in self.ds]):
                    self.padding = [[(dsp - 1) / 2] * 2 for dsp in self.ds]
                else:
                    raise NotImplementedError("Poollayer cannot infer padding for window size "
                                              "{} and stride {}. Please provide padding manually.".format(self.ds,
                                                                                                          self.stride))
        elif isinstance(padding, (tuple, list)):
            self.padding = list(padding)
        else:
            raise NotImplementedError("Padding must be a tuple or a list or keyword 'auto'.")

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

        self.layerinfo = "[{}-pool by {} kernel]".format(self.poolmode, self.ds)

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

        self.y = A.pool(inp=inp, ds=self.ds, stride=self.stride, padding=self.padding, poolmode=self.poolmode,
                        dim=self.dim, ignoreborder=self.ignoreborder, issequence=self.issequence)
        return self.y

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


        usd = A.unpool(inp, us=self.us, interpolate=self.interpolate, dim=self.dim, issequence=self.issequence)

        # This gets ugly: A.unpool does interpolation when us[0] == us[1]
        interpolate = self.interpolate and (self.us[0] != self.us[1] or self.dim == 3)

        # TODO: Move to backend.unpool
        # Interpolate if required
        if interpolate:
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

        self.y = usd
        return self.y


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

        self.y = A.softmax(inp, dim=self.dim, predict=predict, issequence=self.issequence)
        return self.y

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

        elif self.dim == 3:
            # Input
            self.x = T.tensor('floatX', [False, False, False, False, False], name='x:' + str(id(self)))
            # Output
            self.y = T.tensor('floatX', [False, False, False, False, False], name='y:' + str(id(self)))

        elif self.dim == 1:
            # Input
            self.x = T.matrix('x:' + str(id(self)))
            # Output
            self.y = T.matrix('y:' + str(id(self)))

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

        self.y = A.noise(inp, noisetype=self.noisetype, p=self.p, n=self.n, sigma=self.sigma, srng=self.srng)
        return self.y


# Batch Normalization Layer
class batchnormlayer(layer):
    """ Batch Normalization Layer con Nonlinear Activation. See Ioffe et al. (http://arxiv.org/abs/1502.03167). """

    def __init__(self, dim, momentum=0., axis=None, eps=1e-6, activation=netools.linear(gain=1.),
                 gamma=None, beta=None, makedecoder=True, inpshape=None):
        """
        :type dim: int
        :param dim: Dimensionality of the layer/network (2D or 3D)

        :type momentum: float
        :param momentum: Momentum of the moving average and std (over batches)

        :type axis: int
        :param axis: Axis over which to normalize

        :type eps: float
        :param eps: A small epsilon for numerical stability

        :type activation: callable
        :param activation: Activation function.

        :type gamma: callable or float or numpy.ndarray
        :param gamma: Default value for the scale factor gamma. Must be parsable as a ghost variable.

        :type beta: callable or float or numpy.ndarray
        :param beta: Default value for the shift amout beta. Must be parsable as a ghost variable.

        :type makedecoder: bool
        :param makedecoder: Whether to activate decoder

        :type inpshape: tuple or list
        :param inpshape: Expected input shape
        """

        # Init superclass
        super(batchnormlayer, self).__init__()

        # Meta
        self.momentum = momentum
        self.dim = dim
        self.eps = eps
        self.activation = activation
        self.encoderactive = True
        self.decoderactive = makedecoder
        # The input could be a sequence for all I care
        self.allowsequences = True

        # Debug
        self._epreshiftscale = None
        self._epreactivation = None
        self._dpostshiftscale = None
        self._dpreactivation = None

        # Parse input dimensions
        if self.dim == 2:
            self.inpdim = 4
        elif self.dim == 3:
            self.inpdim = 5
        else:
            raise NotImplementedError("Invalid layer dimension. Supported: 2 and 3.")

        # Parse input shape
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim
        else:
            assert len(inpshape) == self.inpdim, "Input shape must have exactly as many elements as the " \
                                                 "input dimension (4 for 2D, 5 for 3D)."
            self.inpshape = inpshape

        # Parse axis
        if axis is None:
            self.axis = {5: 2, 4: 1}[self.inpdim]
        else:
            assert axis in [1, 2]
            self.axis = axis

        # This layer is particularly hairy because the parameter shapes depend on the input shape. Inputs are generally
        # tensor variables while parameters are expected to be shared variables.
        # The user can be expected to provide the input shape of the first layer in a layertrain, but not of a layer
        # deep in the network.
        # Ghost variables to the rescue!
        # Since this is the first time ghost variables are being deployed, here are some general ground rules:
        #   1. The layer must clean up it's own mess: that includes appropriately updating ghost variable shapes and
        #      instantiating new instances of ghost variable when necessary. This prevents the ghost variable mess
        #      from spilling over to other parts of the project.
        #   2. All that can be expected from layertrain (or a more abstract class) is that the instantiate() method of
        #      the ghost variables be called while feeding forward.

        # Get normalization shape

        # Parse default values for ghost params
        if beta is None:
            beta = lambda shp: np.zeros(shape=shp)
        else:
            if callable(beta):
                # Beta is good
                pass
            elif isinstance(beta, float):
                # Convert floating point number to a callable returning number * ones matrix
                beta = (lambda pf: lambda shp: pf * np.ones(shp, dtype=th.config.floatX))(beta)
            else:
                raise NotImplementedError("Beta must be a callable or a floating point number.")

        if gamma is None:
            gamma = lambda shp: np.ones(shape=shp)
        else:
            if callable(gamma):
                # Gamma good
                pass
            elif isinstance(gamma, float):
                gamma = (lambda pf: lambda shp: pf * np.ones(shp, dtype=th.config.floatX))(gamma)
            else:
                raise NotImplementedError("Beta must be a callable or a floating point number.")

        # Function to compute ghost parameter shape given the input shape
        self.getghostparamshape = lambda shp: [shp[self.axis], ]

        # Ghost params
        self.beta = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                      value=beta,
                                      name='bnbeta:' + str(id(self)),
                                      shared=True)
        self.gamma = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                       value=gamma,
                                       name='bngamma:' + str(id(self)),
                                       shared=True)

        # Ghost connectivity params
        self.betac = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                       value=1.,
                                       name='bnbetac:' + str(id(self)),
                                       shared=True)
        self.gammac = netutils.ghostvar(shape=self.getghostparamshape(self.inpshape),
                                        value=1.,
                                        name='bngammac:' + str(id(self)),
                                        shared=True)

        # Gather ghost parameters and cparameters
        self.ghostparams = [self.gamma, self.beta]
        self.ghostcparams = [self.gammac, self.betac]
        # Gather parameters and cparameters. Right now, self.params = self.ghostparams, but that should change in the
        # feedforward() method.
        self._params = [self.gamma, self.beta]
        self._cparams = [self.gammac, self.betac]

        # Initialize state variables
        self.runningmean = 0.
        self.runningstd = 1.

        # Set state
        self.state = [self.runningmean, self.runningstd]

        # Container for input (see feedforward() for input shapes) and output
        # Input
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        # Output
        self.y = T.tensor('floatX', [False, ] * self.outdim, name='y:' + str(id(self)))

    def feedforward(self, inp=None, activation=None):
        # Parse input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Identity if encoder is not active
        if not self.encoderactive:
            return inp

        # Parse activation
        activation = self.activation if activation is None else activation

        # Instantiate params from ghost state
        self.instantiate()

        __old__ = False

        if not __old__:
            if not self.testmode:
                # Training graph
                # Compute batch norm
                y, bm, bstd = A.batchnorm(self.x, self.gamma, self.beta, gammamask=self.gammac, betamask=self.betac,
                                          axis=self.axis, eps=self.eps, dim=self.dim, issequence=self.issequence)
                # Add running mean and running std updates to updaterequests
                updreq = dict(self.updaterequests)
                updreq.update({self.runningmean: self.momentum * self.runningmean + (1 - self.momentum) * bm,
                               self.runningstd: self.momentum * self.runningstd + (1 - self.momentum) * bstd})
                self.updaterequests = updreq.items()

            else:
                y, bm, bstd = A.batchnorm(self.x, self.gamma, self.beta, mean=self.runningmean, std=self.runningstd,
                                          gammamask=self.gammac, betamask=self.betac, axis=self.axis, eps=self.eps,
                                          im=self.dim, issequence=self.issequence)

            self.y = y
            return self.y

        if __old__:
            # Determine reduction axes
            redaxes = range(self.inpdim)
            redaxes.pop(self.axis)
            broadcastaxes = redaxes

            # Compute mean and standard deviation
            batchmean = T.mean(inp, axis=redaxes)
            batchstd = T.sqrt(T.var(inp, axis=redaxes) + self.eps)

            # Broadcast running mean and std, batch mean and std
            broadcastpattern = ['x'] * self.inpdim
            broadcastpattern[self.axis] = 0

            rm = self.runningmean.dimshuffle(*broadcastpattern)
            rstd = self.runningstd.dimshuffle(*broadcastpattern)
            bm = batchmean.dimshuffle(*broadcastpattern)
            bstd = batchstd.dimshuffle(*broadcastpattern)

            if not self.testmode:
                # Place update requests. Remember that feedforward could have been called before (i.e. to use dict updates).
                updreq = dict(self.updaterequests)
                updreq.update({self.runningmean: self.momentum * self.runningmean + (1 - self.momentum) * batchmean,
                               self.runningstd: self.momentum * self.runningstd + (1 - self.momentum) * batchstd})
                self.updaterequests = updreq.items()

                # Normalize input.
                norminp = (inp - bm) / (bstd)
                # For debug
                self._epreshiftscale = norminp
            else:
                norminp = (inp - rm) / (rstd)
                # For debug
                self._epreshiftscale = norminp

            # TODO: Gradient clips
            # Shift and scale
            # Broadcast params
            gammabc = self.gamma.dimshuffle(*broadcastpattern)
            betabc = self.beta.dimshuffle(*broadcastpattern)
            gammacbc = self.gammac.dimshuffle(*broadcastpattern)
            betacbc = self.betac.dimshuffle(*broadcastpattern)

            self._epreactivation = tho.maskgradient(gammabc, gammacbc) * norminp + tho.maskgradient(betabc, betacbc)

            # Activate
            self.y = activation(self._epreactivation)

            # Return
            return self.y

    def instantiate(self):
        # Instantiate with the machinery in layer
        super(batchnormlayer, self).instantiate()

        # Gamma and beta are no longer ghost variables, but real theano variables.
        # Rebind gamma and beta to the instantiated variables
        self.gamma, self.beta = self.params
        self.gammac, self.betac = self.cparams

        # Initialize running mean and running std variables. This should be possible because inpshape is now known.
        self.runningmean = th.shared(np.zeros(shape=self.getghostparamshape(self.inpshape), dtype=th.config.floatX),
                                     name='rmean:' + str(id(self)))
        self.runningstd = th.shared(np.ones(shape=self.getghostparamshape(self.inpshape), dtype=th.config.floatX),
                                    name='rstd:' + str(id(self)))
        # TODO: Have runningmean and runningstd as ghost variables
        self.state = [self.runningmean, self.runningstd]

    def inferoutshape(self, inpshape=None, checkinput=False):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        # Return input shape if encoder not active
        if not self.encoderactive:
            return inpshape

        # Batch normalization does not change the input shape anyway.
        return inpshape
