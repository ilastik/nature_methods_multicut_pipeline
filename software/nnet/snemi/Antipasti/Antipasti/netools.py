import numpy as np
import theano as th
from theano import tensor as T
from theano.tensor import switch, nnet, tanh

import netutils as nu

try:
    import matplotlib.pyplot as plt
except:
    plt = None

__doc__ = \
    """File with activation functions and other general tools required for building a network."""

# Make a callable subclass of dictionary
cdict = type('cdict', (dict,), {'__call__': lambda self, inp: self['function'](inp)})

# Activation Functions
# Rectified Linear
def relu(alpha=0):  # Gradient Tested and Working
    return lambda x: switch(x > 0, x, alpha * x)
    # switch(x < 0, 0, x)


# Truncated Rectified Linear
def trelu(threshold=1, alpha=0):
    return lambda x: switch(x > threshold, x, alpha * x)


# Sigmoid
def sigmoid(gain=1, spread=1, mode='hard'):  # Possible modes: 'fast', 'full', 'hard'.
    if mode == 'fast':
        return lambda x: gain * nnet.ultra_fast_sigmoid(spread * x)
    elif mode == 'full':
        return lambda x: gain * nnet.sigmoid(spread * x)
    elif mode == 'hard':
        return lambda x: gain * nnet.hard_sigmoid(spread * x)
    else:
        return lambda x: gain * nnet.ultra_fast_sigmoid(spread * x)


# Linear (duh)
def linear(gain=1.):
    return lambda x: gain * x


# Truncated Linear
def trlinear(threshold=1, gain=1):
    return lambda x: switch(T.abs_(x) > threshold, gain*x, 0)


# Tanh of ReLU: Saturates at gain for large positive values, gives zero otherwise.
def relutanh(gain=10, spread=0.1):
    return lambda x: gain * tanh(spread * relu()(x))


# Symmetric Linear Unit (SyLU)
def sylu(gain=10, spread=0.1):
    return lambda x: switch(T.ge(x, (1 / spread)), gain, 0) + \
                     switch(T.and_(T.gt((1 / spread), x), T.gt(x, -(1 / spread))), gain * spread * x, 0) + \
                     switch(T.le(x, -(1 / spread)), -gain, 0)


# Exponential Linear Unit
def elu(alpha=1):
    return lambda x: switch(x > 0., x, alpha * (T.exp(x) - 1))


# Make activation functions learnable
def learnable(func, fmapsin=None, a=None, b=None, c=None, d=None):
    """
    Make the activation function `func` learnable, such that func --> c * f(a*x + b) + d

    :type func: callable
    :param func: Activation function (to make learnable)

    :type fmapsin: int
    :param fmapsin: Number of input feature maps. Required if the function is to have independent parameters for every
                   feature map.

    :type a: callable
    :param a: a of func --> c * f(a*x + b) + d.

    :type b: callable
    :param b: b of func --> c * f(a*x + b) + d

    :type c: callable
    :param c: c of func --> c * f(a*x + b) + d

    :type d: callable
    :param d: d of func --> c * f(a*x + b) + d

    :return: c * f(a*x + b) + d with a, b, c, d learnable (if not None)
    """

    # Fmapsin is only required if the function is to have learnable parameters for every feature map. Otherwise, the
    # parameters are broadcasted over the entire stack of feature maps.
    fmapsin = (fmapsin, ) if isinstance(fmapsin, int) else ()

    # Instantiate a list of trainable parameters
    trainables = []

    if a is not None:
        a = nu.getshared(shape=fmapsin, value=a, name='func-a')
        trainables.append(a)
    else:
        a = nu.getshared(shape=fmapsin, value=1.)

    if b is not None:
        b = nu.getshared(shape=fmapsin, value=b, name='func-b')
        trainables.append(b)
    else:
        b = nu.getshared(shape=fmapsin, value=0.)

    if c is not None:
        c = nu.getshared(shape=fmapsin, value=c, name='func-c')
        trainables.append(c)
    else:
        c = nu.getshared(shape=fmapsin, value=1.)

    if d is not None:
        d = nu.getshared(shape=fmapsin, value=d, name='func-d')
        trainables.append(d)
    else:
        d = nu.getshared(shape=fmapsin, value=0.)

    # Define function
    def _func(inp):
        if inp.ndim == 4:
            # Broadcast variables accordingly
            bcst = lambda *variables: [var.dimshuffle('x', 0, 'x', 'x') if var.ndim == 1 else var for var in variables]

        elif inp.ndim == 5:
            # Broadcast variables accordingly
            bcst = lambda *variables: [var.dimshuffle('x', 'x', 0, 'x', 'x') if var.ndim == 1 else var
                                       for var in variables]

        else:
            raise NotImplementedError("Invalid input dimensionality.")

        # Get broadcasted variables
        ba, bb, bc, bd = bcst(a, b, c, d)
        # Make function
        return bc * func(ba * inp + bb) + bd

    # Return a callable dictionary
    return cdict({"function": _func, "trainables": trainables})


# Initializers
# Identity kernel (can be used for Net2DeeperNet)
def idkernel(shape, shared=True):
    # Init a new zero variable of shape shape
    idk = np.zeros(shape=shape, dtype=th.config.floatX)
    # Check if shape corresponds to a 2D or 3D kernel
    if len(shape) == 4:
        # Check if the kernel size is odd or even, and generate appropriate slice objects
        slicelist = [slice(kerdim/2, kerdim/2 + 1) if kerdim % 2 == 1 else slice(kerdim/2 - 1, kerdim/2)
                     for kerdim in shape[2:]]
        # Make identity kernel
        idk[..., slicelist[0], slicelist[1]] = 1.
    elif len(shape) == 5:
        slicelist = [slice(kerdim/2, kerdim/2 + 1) if kerdim % 2 == 1 else slice(kerdim/2 - 1, kerdim/2)
                     for kerdim in shape[1] + shape[3:]]
        # Make identity kernel
        idk[:, slicelist[0], :, slicelist[1], slicelist[2]] = 1

    # Convert to shared variable if requested
    if shared:
        return th.shared(value=idk)
    else:
        return idk


# Initialization schemes
# Glorot Initialization (Normal)
def xavier(shape, dtype=th.config.floatX):
    if len(shape) == 4:
        # 2D convolutions
        # Determine the number of output maps
        fmapsout = shape[0]
        fmapsin = shape[1]
        fov = np.prod(shape[2:])
    elif len(shape) == 5:
        # 3D convolutions
        fmapsout = shape[0]
        fmapsin = shape[2]
        fov = np.prod(shape[3:]) * shape[1]
    else:
        raise NotImplementedError

    # Compute variance for Xavier init
    var = 2./((fmapsin + fmapsout) * fov)
    # Build kernel
    ker = np.random.normal(loc=0., scale=np.sqrt(var), size=tuple(shape)).astype(dtype)
    return ker

# He Initialization
def he(shape, gain=1., dtype=th.config.floatX):
    if len(shape) == 4:
        # 2D convolutions
        fmapsin = shape[1]
        fov = np.prod(shape[2:])
    elif len(shape) == 5:
        # 3D convolutions
        fmapsin = shape[2]
        fov = np.prod(shape[3:]) * shape[1]
    else:
        raise NotImplementedError

    # Parse gain
    if isinstance(gain, str):
        if gain.lower() == 'relu':
            gain = 2.
        elif gain.lower() in ['sigmoid', 'linear', 'tanh']:
            gain = 1.

    # Compute variance for He init
    var = gain/(fmapsin * fov)
    # Build kernel
    ker = np.random.normal(loc=0., scale=np.sqrt(var), size=tuple(shape)).astype(dtype)
    return ker


# Training Monitors
# Batch Number
# Loss
# Cost
# Training Error
# Validation Error
# Gradient Norm Monitor
# Monitor to check if the model is exploring previously unexplored parameter space. Returns n, where n is the number of
# dimensions in which the model has explored new param range.
# Monitor for update norm


if plt is not None:
    # Plotter Function
    pass

