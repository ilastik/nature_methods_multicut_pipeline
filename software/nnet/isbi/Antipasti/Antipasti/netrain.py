__author__ = 'nasimrahaman'

__doc__ = \
    """
    Module to assist in training neural models. Includes cost functions, optimization algorithms.
    Contents:
        Utils:
            prepare data (prep)
        Loss Functions
            cross-entropy loss (ce)
            max-likelihood loss (mll)
            mean squared error loss (mse)
        Regularizers
            Lp term (lp)
        Update Generators
            stochastic gradient descent (sgd)
            adam (adam)
            momentum stochastic gradient descent (momsgd)

    To extend this layer with a(n)
        Loss Function:
            - it must accept ist (source), soll (target). Everything else is optional.
            - it must return a theano scalar.

        Optimizer:
            - it must accept a list of parameters params, cost and/or gradients. Everything else is optional.
            - it must return a list of variable updates, which should look like
              [(oldvar1, newvar1), (oldvar2, newvar2), ...]



    """

import theano as th
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams



# Utils
def prep(ist=None, soll=None, ignoreclass0=False, isautoencoder=False, clip=False):
    """
    Function to prepare ist and soll variables for the selected loss function.

    :type ist: theano.tensor.var.TensorVariable
    :param ist: Source variable (output of the network)

    :type soll: theano.tensor.var.TensorVariable
    :param soll: Target variable (dummy target variable)

    :type ignoreclass0: bool
    :param ignoreclass0: Whether to ignore the zero-th class in ist and soll. Use if the error signals from an auxiliary
                         class (zeroth) are to be discarded (e.g. as not labeled class for image classification).

    :type isautoencoder: bool
    :param isautoencoder: Whether the model is an autoencoder.

    :type clip: bool
    :param clip: Whether to clip ist and soll (recommended, prevents NaNs)

    :return: ist, soll
    """

    # Assertions
    if ist is not None:
        assert ist.ndim in [2, 4, 5], "Source variable (ist) must be 2D (for 1D data) or 4D (for 2D data) or" \
                                      " 5D (for 3D data)"

    if soll is not None:
        assert soll.ndim in [2, 4, 5], "Target variable (soll) must be 2D (for 1D data) or 4D (for 2D data) or" \
                                       " 5D (for 3D data)"

    # For autoencoders: Compare ist and soll image by image.
    # Say ist.shape = soll.shape = (numimg, numchannel, y, x). Flatten to (numimg, numchannel*y*x). Put in another way,
    # pixels are treated as classes.

    # For classifiers: Compare ist and soll class by class, assuming that the channel axis contains class info.
    # Say ist.shape = soll.shape = (numimg, numchannel, y, x). Flatten to (numimg*y*x, numchannel).
    # Pixels are treaded as samples.

    if not isautoencoder:
        # Set up variables for classification cost:
        # If ist or soll is a 2D tensor (for 1D data), assume that the second dimension contains class information
        # and call it a day. Otherwise for 2D or 3D data, the class information is expected in the channel axis:
        # ist.shape = (numimages, numclasses, y, x) in 2D or (numimages, z, numclasses, y, x) in 3D. The first
        # dimshuffle is meant to send the class dimension to first.
        # Now that the first dimension contains class information, the other dimensions are just samples.
        # The aim now is to flatten out all but the first dimension in ist and soll.
        # T.flatten linearizes the second and following dimensions, and the
        # last dimshuffle sends the class dimension (now first) to the back again (as the second dimension).
        # NOTE: Theano variable names are not reused for a good reason.
        if ist is not None:
            if not ist.ndim == 2:
                istf = ist.dimshuffle(*([1, 0, 2, 3] if ist.ndim == 4 else [2, 0, 1, 3, 4])).\
                    flatten(ndim=2).dimshuffle(1, 0)
            else:
                istf = ist
        if soll is not None:
            if not soll.ndim == 2:
                sollf = soll.dimshuffle(*([1, 0, 2, 3] if soll.ndim == 4 else [2, 0, 1, 3, 4])).\
                    flatten(ndim=2).dimshuffle(1, 0)
            else:
                sollf = soll
    else:
        if ist is not None:
            # If model an autoencoder, flatten signal tensor to a signal vector
            # (while leaving the numsignal axis intact).
            if not ist.ndim == 2:
                istf = ist.flatten(ndim=2)
            else:
                istf = ist

        if soll is not None:
            if not soll.ndim == 2:
                sollf = soll.flatten(ndim=2)
            else:
                sollf = soll

    # If class 0 is to be ignored, get rid of it from ist and soll.
    if ignoreclass0 and not isautoencoder:
        # Delete the first column
        if ist is not None:
            istfc = istf[:, 1:]
        if soll is not None:
            sollfc = sollf[:, 1:]
    else:
        # Don't reuse theano variable names!
        if ist is not None:
            istfc = istf
        if soll is not None:
            sollfc = sollf

    if not clip:
        if ist is not None and soll is not None:
            return istfc, sollfc
        if ist is not None:
            return istfc
        if soll is not None:
            return sollfc
    else:
        if ist is not None and soll is not None:
            return T.clip(istfc, 0.01, 0.99), T.clip(sollfc, 0.01, 0.99)
        if ist is not None:
            return T.clip(istfc, 0.01, 0.99)
        if soll is not None:
            return T.clip(sollfc, 0.01, 0.99)


# Loss Functions
def cce(ist, soll, wmap=None, ignoreclass0=False, isautoencoder=False, clip=True):
    """
    Computes the categorical cross-entropy loss. To be used after a sigmoid or softmax layer (i.e. with variables ist
    and soll scaled between 0 and 1) as a classification loss.

    :type ist: theano.tensor.var.TensorVariable
    :param ist: Source variable (output of the network)

    :type soll: theano.tensor.var.TensorVariable
    :param soll: Target variable (dummy target variable)

    :type wmap: theano.tensor.var.TensorVariable
    :param wmap: Weight map (for weighted cross entropy).

    :type ignoreclass0: bool
    :param ignoreclass0: Whether to ignore the zero-th class in ist and soll. Use if the error signals from an auxiliary
                         class (zeroth) are to be discarded (e.g. as not labeled class for image classification).

    :type isautoencoder: bool
    :param isautoencoder: Whether the model is an autoencoder.

    :type clip: bool
    :param clip: Whether to clip ist and soll (recommended, prevents NaNs)

    :return: cost
    """
    # Prep ist and soll variables.
    ist, soll = prep(ist, soll, ignoreclass0=ignoreclass0, isautoencoder=isautoencoder, clip=clip)

    # Prepare weight map
    if wmap is not None:
        wmap = prep(wmap, ignoreclass0=ignoreclass0, isautoencoder=isautoencoder, clip=False)

    if wmap is None:
        # Compute cost
        L = -T.mean(T.sum(soll * T.log(ist), axis=1))
    else:
        L = -T.mean(wmap * T.sum(soll * T.log(ist), axis=1, keepdims=True))

    # Return
    return L


def ce(ist, soll, wmap=None, ignoreclass0=False, isautoencoder=False, clip=True):
    """
    Computes the (binary) cross-entropy loss. To be used after a sigmoid or softmax layer (i.e. with variables ist and
    soll scaled between 0 and 1) as a classification loss.

    :type ist: theano.tensor.var.TensorVariable
    :param ist: Source variable (output of the network)

    :type soll: theano.tensor.var.TensorVariable
    :param soll: Target variable (dummy target variable)

    :type wmap: theano.tensor.var.TensorVariable
    :param wmap: Weight map (for weighted cross entropy).

    :type ignoreclass0: bool
    :param ignoreclass0: Whether to ignore the zero-th class in ist and soll. Use if the error signals from an auxiliary
                         class (zeroth) are to be discarded (e.g. as not labeled class for image classification).

    :type isautoencoder: bool
    :param isautoencoder: Whether the model is an autoencoder.

    :type clip: bool
    :param clip: Whether to clip ist and soll (recommended, prevents NaNs)

    :return: cost
    """
    # Prep ist and soll variables.
    ist, soll = prep(ist, soll, ignoreclass0=ignoreclass0, isautoencoder=isautoencoder, clip=clip)

    # Prepare weight map
    if wmap is not None:
        wmap = prep(wmap, ignoreclass0=ignoreclass0, isautoencoder=isautoencoder, clip=False)

    if wmap is None:
        # Compute cost
        L = -T.mean(T.sum(soll * T.log(ist) + (1. - soll) * T.log(1. - ist), axis=1))
    else:
        L = -T.mean(wmap * T.sum(soll * T.log(ist) + (1. - soll) * T.log(1. - ist), axis=1, keepdims=True))

    # Return
    return L


def mll(ist, soll, ignoreclass0=False, isautoencoder=False, clip=True):
    """
    Function to compute the maximum log likelihood loss.

    :type ist: theano.tensor.var.TensorVariable
    :param ist: Source variable (output of the network)

    :type soll: theano.tensor.var.TensorVariable
    :param soll: Target variable (dummy target variable)

    :type ignoreclass0: bool
    :param ignoreclass0: Whether to ignore the zero-th class in ist and soll. Use if the error signals from an auxiliary
                         class (zeroth) are to be discarded (e.g. as not labeled class for image classification).

    :type isautoencoder: bool
    :param isautoencoder: Whether the model is an autoencoder.

    :type clip: bool
    :param clip: Whether to clip ist and soll (recommended, prevents NaNs)

    :return: loss
    """

    # Prep ist and soll variables.
    ist, soll = prep(ist, soll, ignoreclass0=ignoreclass0, isautoencoder=isautoencoder, clip=clip)

    # Compute cost
    L = -T.mean(T.log(T.sum(ist * soll, axis=1)))

    # Return
    return L


def mse(ist, soll, ignoreclass0=False, isautoencoder=False):
    """
    Function to compute mean squared error loss. To be used for regression.

    :type ist: theano.tensor.var.TensorVariable
    :param ist: Source variable (output of the network)

    :type soll: theano.tensor.var.TensorVariable
    :param soll: Target variable (dummy target variable)

    :type ignoreclass0: bool
    :param ignoreclass0: Whether to ignore the zero-th class in ist and soll. Use if the error signals from an auxiliary
                         class (zeroth) are to be discarded (e.g. as not labeled class for image classification).

    :type isautoencoder: bool
    :param isautoencoder: Whether the model is an autoencoder.

    :return: loss
    """

    # Prep ist and soll variables.
    ist, soll = prep(ist, soll, ignoreclass0=ignoreclass0, isautoencoder=isautoencoder)

    # Compute cost
    L = T.mean(T.sum((ist - soll) ** 2, axis=1))

    # Return
    return L


# Regularizers
def lp(params, regterms=[]):
    """
    Compute the sum of multiple Lp norms.

    :type params: list
    :param params: Parameters to take the Lp norm of.

    :type regterms: list
    :param regterms: List of Lp terms with lambdas. E.g.: [(1, 0.01), (2, 0.001)] <--> 0.01 * L1 norm + 0.001 * L2 norm
    """
    return sum([lamb * Lp(params=params, p=p) for p, lamb in regterms])


def Lp(params, p=2):
    """
    Given a list of parameters, compute the p-th power of its Lp norm.

    :type params: list
    :param params: Parameters to take the Lp norm of.

    :type p: int
    :param p: p of the Lp norm. Defaults to 2.

    :return: (Lp norm)^p
    """

    # Compute Lp^p
    lpn = sum(map(T.sum, map(lambda k: k ** p, params)))

    # Return
    return lpn


# Update Generators
def sgd(params, cost=None, gradients=None, learningrate=1e-4):
    """
    Computes the updates for Stochastic Gradient Descent (without momentum)

    :type params: list
    :param params: Network parameters.

    :type cost: theano.tensor.var.TensorVariable
    :param cost: Cost variable (scalar). Optional if the gradient is provided.

    :type gradients: list
    :param gradients: Gradient of a cost w.r.t. parameters. Optional if the cost is provided.

    :type learningrate: theano.tensor.var.TensorVariable or float
    :param learningrate: Learning rate of SGD. Can be a float (static) or a dynamic theano variable.

    :return: List of updates
    """

    # Validate input
    assert not (cost is None and gradients is None), "Update function sgd requires either a cost scalar or a list of " \
                                                     "gradients."

    # Compute gradients if requested
    if gradients is None and cost is not None:
        pdC = T.grad(cost, wrt=params)
        # Kill gradients if cost is nan
        dC = [th.ifelse.ifelse(T.isnan(cost), T.zeros_like(dparam), dparam) for dparam in pdC]
    else:
        dC = gradients

    # Compute updates
    upd = [(param, param - learningrate * dparam) for param, dparam in zip(params, dC)]

    # Return
    return upd


# ADAM
def adam(params, cost=None, gradients=None, learningrate=0.0002, beta1=0.9, beta2=0.999, epsilon=1e-8, eta=0.,
         gamma=0.55, iterstart=0):
    """
    Computes the updates for ADAM.

    :type params: list
    :param params: Network parameters.

    :type cost: theano.tensor.var.TensorVariable
    :param cost: Cost variable (scalar). Optional if the gradient is provided.

    :type gradients: list
    :param gradients: Gradient of a cost w.r.t. parameters. Optional if the cost is provided.

    :type learningrate: theano.tensor.var.TensorVariable or float
    :param learningrate: Learning rate of SGD. Can be a float (static) or a dynamic theano variable.

    :type beta1: float
    :param beta1: See Kingma and Ba 2014: http://arxiv.org/abs/1412.6980

    :type beta2: float
    :param beta2: See Kingma and Ba 2014: http://arxiv.org/abs/1412.6980

    :type epsilon: float
    :param epsilon: See Kingma and Ba 2014: http://arxiv.org/abs/1412.6980

    :type eta: float
    :param eta: Eta for noisy gradient. See Neelakantan et al. 2015: http://arxiv.org/pdf/1511.06807v1.pdf

    :type gamma: float
    :param gamma: Gamma for noisy gradient. See Neelakantan et al. 2015: http://arxiv.org/pdf/1511.06807v1.pdf

    :type iterstart: int or float
    :param iterstart: Adam anneals the learning rate with iterations. This parameter specifies the initial value of the
                      iteration count, such that the learning rate is scaled appropriately (or the model might jump out
                      of the potential minimum where it's at).

    :return: List of updates
    """

    # Validate input
    assert not (cost is None and gradients is None), "Update function adam requires either a cost scalar or a list of " \
                                                     "gradients."

    # Compute gradients if requested
    if gradients is None and cost is not None:
        pdC = T.grad(cost, wrt=params)
        # Kill gradients if cost is nan
        dC = [th.ifelse.ifelse(T.isnan(cost), T.zeros_like(dparam), dparam) for dparam in pdC]
    else:
        dC = gradients

    updates = []

    # Gradient noising
    if not (eta == 0):
        # RNG
        srng = RandomStreams()
        # Iteration counter
        itercount = th.shared(np.asarray(iterstart, dtype=th.config.floatX))
        # Add noise
        dC = [dparam + srng.normal(size=dparam.shape, std=T.sqrt(eta/(1 + itercount)**gamma), dtype='floatX')
              for dparam in dC]
        # Update itercount
        updates.append((itercount, itercount + 1))

    # Implementation as in reference paper, nothing spectacular here...
    tm1 = th.shared(np.asarray(iterstart, dtype=th.config.floatX))
    t = tm1 + 1
    at = learningrate*T.sqrt(1-beta2**t)/(1-beta1**t)

    for param, dparam in zip(params, dC):
        paramshape = param.get_value().shape

        mtm1 = th.shared(np.zeros(paramshape, dtype=th.config.floatX))
        vtm1 = th.shared(np.zeros(paramshape, dtype=th.config.floatX))

        mt = beta1 * mtm1 + (1 - beta1) * dparam
        vt = beta2 * vtm1 + (1 - beta2) * dparam**2
        u = at * mt/(T.sqrt(vt) + epsilon)

        updates.append((mtm1, mt))
        updates.append((vtm1, vt))
        updates.append((param, param - u))

    updates.append((tm1, t))

    return updates


# Momentum SGD
def momsgd(params, cost=None, gradients=None, learningrate=0.01, momentum=0.9, nesterov=True):
    # TODO: Docstring
    # Validate input
    assert not (cost is None and gradients is None), "Update function momsgd requires either a cost scalar or a " \
                                                     "list of gradients."

    # Compute gradients if requested
    if gradients is None and cost is not None:
        pdC = T.grad(cost, wrt=params)
        # Kill gradients if cost is nan
        dC = [th.ifelse.ifelse(T.isnan(cost), T.zeros_like(dparam), dparam) for dparam in pdC]
    else:
        dC = gradients

    # Init update list
    updates = []

    for param, dparam in zip(params, dC):
        # Fetch parameter shape
        paramshape = param.get_value().shape
        # ... and init initial momentum
        mom = th.shared(np.zeros(paramshape, dtype=th.config.floatX))
        # Compute velocity
        vel = momentum * mom - learningrate * dparam

        # Compute new parameters
        if nesterov:
            newparam = param + momentum * vel - learningrate * dparam
        else:
            newparam = param + vel

        # update update list
        updates.append((param, newparam))
        updates.append((mom, vel))

    # Return
    return updates


def rmsprop(params, cost=None, gradients=None, learningrate=0.0005, rho=0.9, epsilon=1e-6):

    # Validate input
    assert not (cost is None and gradients is None), "Update function rmsprop requires either a cost scalar or a " \
                                                     "list of gradients."

    # Compute gradients if requested
    if gradients is None and cost is not None:
        pdC = T.grad(cost, wrt=params)
        # Kill gradients if cost is nan
        dC = [th.ifelse.ifelse(T.isnan(cost), T.zeros_like(dparam), dparam) for dparam in pdC]
    else:
        dC = gradients

    # Init update list
    updates = []

    for p, g in zip(params, dC):
        acc = th.shared(p.get_value() * 0.)
        newacc = rho * acc + (1 - rho) * g ** 2
        gradscale = T.sqrt(newacc + epsilon)
        g = g / gradscale
        updates.append((acc, newacc))
        updates.append((p, p - learningrate * g))

    return updates

# Aliases
bce = ce
