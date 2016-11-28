import theano as th
from theano import tensor as T
from theano.tensor import switch, nnet, tanh
import numpy as np
import netutils as nu
try:
    import matplotlib.pyplot as plt
except:
    plt = None

__doc__ = \
    """File with activation functions and other general tools required for building a network."""


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
def batchmonitor(batchnum=None):
    if batchnum is not None:
        return "| Batch: {} |".format(batchnum)
    else:
        return ""


# Loss
def lossmonitor(loss=None):
    if loss is not None:
        return "| Loss: {0:.7f} |".format(float(loss))
    else:
        return ""


# Cost
def costmonitor(cost=None):
    if cost is not None:
        return "| Cost: {0:.7f} |".format(float(cost))
    else:
        return ""


# Training Error
def trEmonitor(trE=None):
    if trE is not None:
        return "| Training Error: {0:.3f} |".format(float(trE))
    else:
        return ""


# Validation Error
def vaEmonitor(vaE=None):
    if vaE is not None:
        return "| Validation Error: {0:.3f} |".format(float(vaE))
    else:
        return ""


# Gradient Norm Monitor
def gradnormmonitor(dC=None):
    if dC is not None:
        gradnorm = sum([np.sum(dparam ** 2) for dparam in dC])
        return "| Gradient Norm: {0:.7f} |".format(float(gradnorm))
    else:
        return ""


# Monitor to check if the model is exploring previously unexplored parameter space. Returns n, where n is the number of
# dimensions in which the model has explored new param range.
def exploremonitor(params=None):
    if hasattr(exploremonitor, "prevparams"):
        # Convert params to numerical value
        params = nu.sym2num(params)
        # Compute number of new dimensions explored
        newdims = sum([np.count_nonzero(param > ubparam) + np.count_nonzero(param < lbparam)
                       for param, ubparam, lbparam in zip(params, exploremonitor.ub, exploremonitor.lb)])
        # Update lower and upper bounds
        exploremonitor.lb = [np.minimum(prevparam, param)
                             for prevparam, param in zip(exploremonitor.prevparams, params)]
        exploremonitor.ub = [np.maximum(prevparam, param)
                             for prevparam, param in zip(exploremonitor.prevparams, params)]
        # Update previous params
        exploremonitor.prevparams = params
        # Compute the total number of parameters
        numparams = sum([np.size(param) for param in params])
        # Compute the volume of the quadratic hull
        hullvol = np.prod(np.array([np.prod(np.abs(ubparam - lbparam))
                                    for ubparam, lbparam in zip(exploremonitor.ub, exploremonitor.lb)]))
        # Return
        return "| Explored Dimensions: {} of {} || Hull Volume: {} |".format(int(newdims),
                                                                             int(numparams),
                                                                             float(hullvol))
    else:
        # Log params
        exploremonitor.prevparams = nu.sym2num(params)
        # Log lower and upper bounds
        exploremonitor.lb = exploremonitor.prevparams
        exploremonitor.ub = exploremonitor.prevparams
        # Return
        return "| Explored Dimensions: N/A |"


# Monitor for update norm
def updatenormmonitor(params):
    if hasattr(updatenormmonitor, "prevparams"):
        # Convert to num
        params = nu.sym2num(params)
        # Compute update norm
        updnorm = sum([np.sum((currparam - prevparam) ** 2)
                       for currparam, prevparam in zip(params, updatenormmonitor.prevparams)])
        # Log
        updatenormmonitor.prevparams = params
        return "| Update Norm: {0:.7f} |".format(float(updnorm))
    else:
        updatenormmonitor.prevparams = nu.sym2num(params)
        return "| Update Norm: N/A |"
    pass


if plt is not None:
    # Plotter Function
    def plotmonitor(trE=None):
        # Make a call counter (to count the number of times the function was called)
        if not hasattr(plotmonitor, "callcount"):
            plotmonitor.callcount = 0
        else:
            plotmonitor.callcount += 1

        # Make a variable to store a history of input arguments
        if not hasattr(plotmonitor, "history"):
            plotmonitor.history = np.array([])
            plotmonitor.history = np.append(plotmonitor.history, trE)
        else:
            plotmonitor.history = np.append(plotmonitor.history, trE)

        # Plot
        # Make x and y values
        x = np.array(range(plotmonitor.callcount + 1))
        y = plotmonitor.history
        if not hasattr(plotmonitor, "fig"):
            plotmonitor.fig = plt.figure()
            plotmonitor.ax = plotmonitor.fig.add_subplot(111)
            plotmonitor.line1, = plotmonitor.ax.plot(x, y, 'r-')
        else:
            plotmonitor.line1.set_ydata(y)
            plotmonitor.line1.set_xdata(x)

        return ""

