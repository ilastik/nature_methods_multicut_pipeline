__author__ = 'nasimrahaman'

''' Neural Network Utilities '''

# Contents:
#   RELU (Theano Symbolic)
#   plot2dconvfilters

# Imports
from warnings import warn

import numpy as np
import theano as th
import theano.tensor as T

import inspect
import pykit as pyk


# Function to read numerical values from theano shared and allocated variables
def sym2num(value):
    # Make sure value is a list
    if not isinstance(value, list):
        value = [value]
    # Make sure all elements in value are numerical (numpy ndarrays)
    try:
        value = [(param.get_value() if not isinstance(param, np.ndarray) else param) for param in value]
    except AttributeError:
        value = [(param.eval() if not isinstance(param, np.ndarray) else param) for param in value]

    return value


# Function to get shared variables of given shape (or the shape of a given variable) filled with a given value.
def getshared(like=None, shape=None, value=0., name=None):
    """
    Returns a shared variable like a given model or of a given shape filled with some value.

    :type like: theano.tensor.sharedvar.TensorSharedVariable or numpy.ndarray
    :param like: Variable to get shape from. Optional if shape is given.

    :type shape: list or tuple
    :param shape: Shape of the shared variable

    :type value: float or int
    :param value: Value to fill the shared variable with

    :type name: str
    :param name: Name of the shared variable

    :rtype: theano.tensor.sharedvar.TensorSharedVariable
    """

    # Assertions
    assert not (like is None and shape is None), "A model variable or a shape must be provided."

    # Compute shape if not given
    try:
        # We could use like.shape.eval() to avoid the try clause, but that might slow things down
        shape = like.get_value().shape if not isinstance(like, np.ndarray) else like.shape if shape is None else shape
    except AttributeError:
        shape = like.eval().shape if not isinstance(like, np.ndarray) else like.shape if shape is None else shape

    # Build the numpy variable for the shared constructor
    numvalue = value * np.ones(shape=shape, dtype=th.config.floatX) \
        if value is not 0. else np.zeros(shape=shape, dtype=th.config.floatX)

    # Build shared variable and return
    return th.shared(value=numvalue, name=name)


# Function to check if a given function has a given kwarg(s)
def haskwarg(fun, *funkwargs):
    # Check
    haskw = [kw in inspect.getargspec(fun).args for kw in funkwargs]
    # Get rid of the list wrap if funkwargs only has one element
    haskw = haskw[0] if len(haskw) == 1 else haskw
    return haskw


# Function to make a given function work with overcomplete kwargs (i.e. accept correct kwargs, ignore the rest)
def smartfunc(fun, ignorekwargssilently=False):

    def smart(func, *args, **kwargs):
        # Check if func is a callable object, in which case extract the call method
        func = func if inspect.isfunction(func) else func.__call__
        # Fetch the list of keywords of the function
        funkws = inspect.getargspec(func).args
        # Instantiate a dictionary to send the function
        sendkwargs = {}
        # Loop over kwargs and select the ones that are required by the function (i.e. are in funkws)
        for kw in kwargs.keys():
            if kw in funkws:
                sendkwargs.update({kw: kwargs[kw]})

        # Compare arguments and raise warning if requested
        if not ignorekwargssilently:
            ignoredkwargs = list(set(kwargs.keys()) - set(sendkwargs.keys()))
            if len(ignoredkwargs) is not 0:
                print("Smart function {} ignores the following parameters: {}".format(func.__name__, ignoredkwargs))

        # Call function and return
        return func(*args, **sendkwargs)

    return lambda *args, **kwargs: smart(fun, *args, **kwargs)


# Function to vectorize a function (i.e. make it work on list)
def vectorizefunc(fun):

    def func(*args, **kwargs):
        # Compute argument length
        arglen = max([pyk.smartlen(arg) for arg in list(args) + list(kwargs.values())])

        # Convert arguments to list
        args = [pyk.obj2list(arg) if not len(pyk.obj2list(arg)) == 1 else pyk.obj2list(arg)*arglen for arg in args]
        kwargs = {kw: pyk.obj2list(arg) if not len(pyk.obj2list(arg)) == 1 else pyk.obj2list(arg)*arglen
                  for kw, arg in kwargs.items()}

        # Make sure list sizes check out
        assert all([pyk.smartlen(arg) == arglen for arg in args]) if pyk.smartlen(arg) != 0 else True, \
            "Input lists must all have the same length."
        assert all([pyk.smartlen(kwarg) == pyk.smartlen(kwargs.values()[0]) == arglen for kwarg in kwargs.values()]) \
            if pyk.smartlen(kwargs) != 0 else True, "Keyword argument vectors must have the same length (= argument " \
                                                    "vector length)"

        # Run the loop (can be done with a long-ass list comprehension, but I don't see the point)
        res = []
        if not len(kwargs) == 0 and not len(args) == 0:
            for arg, kwarg in zip(zip(*args), zip(*kwargs.values())):
                res.append(fun(*arg, **{kw: kwa for kw, kwa in zip(kwargs.keys(), kwarg)}))
        else:
            if len(kwargs) == 0:
                res = [fun(*arg) for arg in zip(*args)]
            elif len(args) == 0:
                res = [fun(**{kw: kwa for kw, kwa in zip(kwargs.keys(), kwarg)}) for kwarg in zip(*kwargs.values())]
            else:
                return []

        # Return results
        return pyk.delist(res)

    return func


# Function to split keyword arguments
def splitkwargs(allkwargs, *kwms):
    """
    Splits dict of keywords according to markers in kwms. E.g. If sep = ':' and kwms = ['loss', 'regularizer'],
    allwkwargs = {'loss:momentum': 0.9, 'loss:learningrate': 0.01, 'regularizer:p', 2} will be split to two dicts:
    {'momentum': 0.9, 'learningrate': 0.01}, {'p', 2}.

    :type allkwargs: dict
    :param allkwargs: key word arguments to be processed.

    :type kwms: list
    :param kwms: List of key word markers to look for.

    :rtype: tuple
    """

    # Fetch keys
    allkws = allkwargs.keys()

    # Parse
    if all([':' in kw for kw in allkws]):
        # Split by colon ':'
        allkwssplit = zip(*[kw.split(':') for kw in allkws])
    else:
        assert not any([':' in kw for kw in allkws]), "You either provide keyword markers or you don't, but not both."
        # Replicate allkwargs multiple times (as required by the number of given kwms) and return
        return [allkwargs for _ in kwms]

    # Build a list to output
    # Prototype 1-liner
    # outlist = [{allkwssplit[1][kwargnum]: allkwargs.items()[kwargnum]
    #            for kwargnum, kwminkwargs in enumerate(allkwssplit[0]) if kwm is kwminkwargs} for kwm in kwms]
    # Better readability:
    outlist = []
    for kwm in kwms:
        markerdict = {}
        for kwargnum, kwminkwargs in enumerate(allkwssplit[0]):
            if kwm == kwminkwargs:
                markerdict.update({allkwssplit[1][kwargnum]: allkwargs.values()[kwargnum]})
        outlist.append(markerdict)

    # Replace empty dictionaries with allkwargs. Marker dictionaries are empty when keyword markers in kwms are not
    # found in kwargs, in which case the corresponding kwarg dict is set to the entire kwarg dict
    outlist = [dict(zip(allkwssplit[1], allkwargs.values())) if len(mdict.items()) is 0 else mdict for mdict in outlist]

    return tuple(outlist)


# Ghost variable class
class ghostvar:
    """
    Ghost variables are theano variables which cannot be immediately initialized because only a subset of their shape
    vector is known.
    """
    def __init__(self, shape=None, value=None, shared=False, name=None):
        """
        :type shape: None or list
        :param shape: Shape of the ghost variable. E.g.: [None, 2, None, None]

        :type value: float or int or callable
        :param value: Value of the ghost variable. Could be a number, in which case the variable is number * ones(shape)
                      Could also be a callable, in which case it gets called with the shape value (for shared vars).

        :type shared: bool
        :param shared: Whether the variable initialized should be shared

        :type name: str
        :param name: Theano variable name.
        """

        # Parse number of dimensions shape, and whether output is shared
        self.ndim = len(shape) if shape is not None else 0
        self._shape = shape
        self.shared = shared
        self.name = name

        # Parse value. Default to 0 if none given:
        if value is None:
            self.value = 0
        else:
            self.value = value

    # Decide if variable is instantiable with the given shape (i.e. if shape is omitted in the instantiate method)
    @property
    def instantiable(self):
        return all([shp is not None for shp in self.shape]) if self.shape is not None else False

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        self.ndim = len(value) if value is not None else 0

    def instantiate(self, shape=None):
        # Parse shape
        shape = [None, ] * self.ndim if shape is None else shape
        initshape = tuple([shape[n] if givenshape is None else givenshape for n, givenshape in enumerate(self.shape)])
        assert all([ishp is not None for ishp in initshape]), "Given shape information not sufficient to instantiate " \
                                                              "from ghost state."

        # Initialize. If shape is a tensor variable, initialize a tensor variable and return.
        if isinstance(shape, T.vector().__class__) or not self.shared:
            # Make variable
            var = T.zeros(shape=initshape, dtype='floatX') \
                if self.value == 0. else self.value(initshape) * T.ones(shape=initshape, dtype='floatX') \
                if callable(self.value) else self.value * T.ones(shape=initshape, dtype='floatX')
            # Safety cast
            var = T.cast(var, dtype='floatX')
            var.name = self.name
            # Warn if a shared variable is requested
            if self.shared:
                warn("Provided shape variable is a theano tensor variable, it cannot be used to initialize a shared "
                     "variable.")
            # Return
            return var
        else:
            # Make variable
            var = th.shared((getattr(np, th.config.floatX)(self.value)
                             if not callable(self.value) and not np.isscalar(self.value) else
                             getattr(np, th.config.floatX)(self.value(initshape)) if callable(self.value) else
                             self.value * np.ones(shape=initshape, dtype=th.config.floatX)))

            var.name = self.name
            # Safety cast and return
            return var

    # Methods to make ghost variables a little more compatible with theano (shared) variables
    # Set value method
    def set_value(self, value):
        self.value = value
        # Reset shape if value is a numpy array
        if isinstance(value, np.ndarray):
            self.shape = value.shape

    # Get value methods
    def get_value(self):
        return self.value

    def eval(self):
        return self.value

    def __repr__(self):
        return "ghost-" + (self.name if self.name is not None else "variable")


# Circuit for a feedforward layer
def fflayercircuit():
    # Init a null circuit
    c = np.zeros(shape=(4, 2, 2))
    # Circuit defaults
    c[0, 0, 0] = 1
    c[1, 0, 1] = 1
    c[2, 0, 1] = 1
    c[3, 1, 1] = 1
    # Ret
    return c


# Set "matrix elements" in a convolution kernel to a specific value
def setkernel(inds, subkernel, out):

    # Take measurements
    kershape = out.shape

    # out can be (fmapsout, fmapsin, kr, kc) or (fmapsout, kT, fmapsin, kr, kc). In case it's the latter, reshape to
    # (fmapsout, fmapsin, kT, kr, kc)
    if out.ndim == 5:
        reshaped = True
        out = np.swapaxes(out, 1, 2)
    else:
        reshaped = False

    # Broadcast subkernel if required
    subkernel = subkernel * np.ones_like(out[0, 0, ...]) if np.size(subkernel) == 1 else subkernel

    for ind in inds:
        out[ind[0], ind[1], ...] = subkernel

    # Reshape back to orig shape
    out = np.swapaxes(out, 1, 2) if reshaped else out

    # Ret
    return out


# Function to flush the value of a shared variable
def flushshared(var, fillfunc=lambda shp: np.random.uniform(size=shp)):
    # Fetch variable value
    varval = var.get_value()
    # Compute new variable value
    newvarval = fillfunc(varval.shape)
    # Set new variale value
    var.set_value(newvarval)
    # Return
    return var


# Function to check if a variable is shared
def isshared(var):
    # This is not as straight forward as with isinstance in Theano.
    shared = hasattr(var, 'get_value') and hasattr(var, 'set_value') and not isinstance(var, ghostvar)
    return shared


# Function to pad theano tensors
def pad(inp, padding):

    if all([padval == 0 for padval in pyk.flatten(padding)]):
        return inp

    if inp.ndim == 4:
        # Make a zero tensor of the right shape
        zt = T.zeros(shape=(inp.shape[0], inp.shape[1], inp.shape[2]+sum(padding[0]), inp.shape[3]+sum(padding[1])))
        # Compute assignment slice
        [[ystart, ystop], [xstart, xstop]] = [[padval[0], (-padval[1] if padval[1] != 0 else None)]
                                              for padval in padding]
        # Assign subtensor
        padded = T.set_subtensor(zt[:, :, ystart:ystop, xstart:xstop], inp)
        return padded
    elif inp.ndim == 5:

        # Make a zero tensor of the right shape
        zt = T.zeros(shape=(inp.shape[0], inp.shape[1]+sum(padding[2]), inp.shape[2], inp.shape[3]+sum(padding[0]),
                            inp.shape[4]+sum(padding[1])))
        # Compute assignment slice
        [[ystart, ystop], [xstart, xstop], [zstart, zstop]] = [[padval[0], (-padval[1] if padval[1] != 0 else None)]
                                                               for padval in padding]
        # Assign subtensor
        padded = T.set_subtensor(zt[:, zstart:zstop, :, ystart:ystop, xstart:xstop], inp)
        return padded
    else:
        raise NotImplementedError("Padding is only implemented for 4 and 5 dimensional tensors.")


# Function to eval and test a (sub-)model for correctness (at forward pass)
def testmodel(model, inpshape, verbose=True, outshape=None):

    testpassed = None

    # Allocate numerical input values (model may have multiple inputs/outputs)
    inpvals = [np.random.uniform(size=ishp).astype(th.config.floatX) for ishp in pyk.list2listoflists(inpshape)]

    # Feedforward the model (i.e. build theano graph if it isn't built already)
    if any([y.owner is None for y in pyk.obj2list(model.y)]):
        model.feedforward()

    # Evaluate model for all inputs
    try:
        if verbose:
            print("Compiling...")

        outvals = [y.eval({x: xval for x, xval in zip(pyk.obj2list(model.x), inpvals)}) for y in pyk.obj2list(model.y)]

        if verbose:
            print("Output shapes are: {}".format([oval.shape for oval in outvals]))

    except Exception as e:
        testpassed = False
        if verbose:
            print("Test failed because an exception was raised. The original error message follows:")
            print(e.message)

    # Check if output shapes check-out
    if outshape is not None and testpassed is None:
        modeloutshape = [oval.shape for oval in outvals]
        shapecheck = modeloutshape == pyk.list2listoflists(outshape)
        if not shapecheck:
            if verbose:
                "Shape check failed. Expected output shape {}, got {} instead.".format(outshape,
                                                                                       pyk.delist(modeloutshape))
            testpassed = False

    if testpassed is None:
        testpassed = True

    return testpassed


# Function to count the number of parameters in a model
def countparams(model):
    return sum([param.get_value().size for param in model.params])
