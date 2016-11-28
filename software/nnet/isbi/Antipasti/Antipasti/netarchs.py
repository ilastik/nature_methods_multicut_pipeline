from Antipasti import netutils

__author__ = "nasimrahaman"

''' Network architectures built on Netkit '''

import numpy as np
import scipy.spatial.distance as ssd
import theano as th
import theano.tensor as T
import netkit as nk
import netrain as nt
import netools as ntl
import netdatakit as ndk
import pykit as pyk
import cPickle as pkl
import os
import datetime
import copy
import time
import re
from warnings import warn


class model(object):
    """ Class to wrap neural models. """

    def __init__(self):
        """
        This function initializes abstract fields.
        """
        # Init with duck typed params
        self._params = []
        self._cparams = []
        self._inpshape = None
        self.updaterequests = []
        self.outshape = None

        self.numinp = None
        self.numout = None

        # I/O
        self.lastsavelocation = None
        self.savedir = None

        self.x = None
        self.y = None
        self.yt = None
        self.xr = None

        # Initialize trainer parameters
        self.C = T.scalar('model-C:' + str(hash(self)), dtype='floatX')
        self.L = T.scalar('model-L:' + str(hash(self)), dtype='floatX')
        self.E = T.scalar('model-E:' + str(hash(self)), dtype='floatX')
        self.dC = []
        self.updates = []
        self.isautoencoder = None

        # Cache compiled functions
        self.classifiertrainer = None
        self.autoencodertrainer = None
        self.classifier = None
        self.autoencoder = None
        self.classifiererror = None
        self.reconstructionerror = None

    # Duck-typed property getter and setters
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        # Convert to list
        value = pyk.obj2list(value)
        # Convert to numeric
        value = netutils.sym2num(value)
        # Apply parameters
        self.applyparams(params=value)

    @property
    def cparams(self):
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        # Convert to list
        value = pyk.obj2list(value)
        # Convert to numeric
        value = netutils.sym2num(value)
        # Apply parameters
        self.applyparams(cparams=value)

    @property
    def inpshape(self):
        return self._inpshape

    @inpshape.setter
    def inpshape(self, value):
        self._inpshape = value

    # Feedforward method
    def feedforward(self, inp=None):
        """
        This builds the theano graph connecting the model input (x) to model output (y)
        :param inp: external input (optional).
        """
        # Return input
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        self.y = inp
        return inp

    # Decoder feedforward model
    def decoderfeedforward(self, inp=None):
        """
        This builds the theano graph connecting the model output (y) to model reconstruction (xr)
        :param inp: external input (optional).
        """
        # Return input
        if inp is None:
            inp = self.y
        else:
            self.y = inp

        self.xr = inp
        return inp

    # Method to apply parameters
    def applyparams(self, params=None, cparams=None):
        """
        Method to apply parameters and connectivity masks.

        :type params: list
        :param params: parameters

        :type cparams: list
        :param cparams: connectivity masks
        """
        pass

    # Method to infer output shape
    def inferoutshape(self, inpshape=None):
        """
        Component for auto shape inference.

        :type inpshape: list
        :param inpshape: Shape of the model input
        """
        if inpshape is None:
            inpshape = self.inpshape
        return inpshape

    # Method to add models
    def __add__(self, other):
        """
        Addition method to stack models (in depth)

        :type other: model or netkit.layer
        :param other: Summand (second model to be added)
        """
        raise NotImplementedError("Addition not implemented for this model.")

    # Method to compute inference error
    def error(self, ist=None, soll=None, ignoreclass0=False, modeltype='classification'):
        """
        Method to compute inference error.

        :type ist: theano.tensor.var.TensorVariable
        :param ist: Source variable

        :type soll: theano.tensor.var.TensorVariable
        :param soll: Target variable

        :type ignoreclass0: bool
        :param ignoreclass0: Whether to ignore the 0-th class while computing error.

        :type modeltype: str
        :param modeltype: Model type. Possible keys: 'classification' or 'regression'
        """

        # Parse errors

        if ist is None:
            if not self.isautoencoder:
                # Feedforward if not done by the user
                if None in [y.owner for y in pyk.obj2list(self.y)]:
                    self.feedforward()
                ist = self.y
            else:
                if None in [y.owner for y in pyk.obj2list(self.xr)]:
                    if None in [y.owner for y in pyk.obj2list(self.y)]:
                        self.feedforward()
                    self.decoderfeedforward()
                ist = self.xr
        if soll is None:
            soll = self.yt if not self.isautoencoder else self.x

        # Prepare data
        pist, psoll = netutils.vectorizefunc(nt.prep)(ist, soll, isautoencoder=self.isautoencoder,
                                                      ignoreclass0=ignoreclass0)

        # Check if model is a classifier or a regression
        isclassifier = (not self.isautoencoder) and (modeltype in ['classifier', 'classification'])

        # pist and psoll are now 2D arrays with class information along the second axis. Flatten them both to a vector
        # using argmax along the class axis
        if isclassifier:
            cist, csoll = netutils.vectorizefunc(T.argmax)(pist, axis=1), \
                          netutils.vectorizefunc(T.argmax)(psoll, axis=1)
        else:
            cist, csoll = pist, psoll

        # Compute error and assign. For classifiers, compute the average number of misses; for autoencoders, compute
        # rms reconstruction error.
        err = netutils.vectorizefunc(lambda i, s: T.mean(T.neq(i, s)))(cist, csoll) if isclassifier else \
              netutils.vectorizefunc(lambda i, s: T.sqrt(T.mean((i - s) ** 2)))(cist, csoll)

        # Add up errors
        err = sum(pyk.obj2list(err))

        self.E = err

        # Return
        return err

    # Method to compute cost
    def cost(self, ist=None, soll=None, params=None, method='ce', isautoencoder=False, regularizer='lp', vectorize=True,
             nanguard=True, **kwargs):
        """
        Computes the cost given a method, source (ist) and target (soll). Regularizers can also be specified.

        :type ist: theano.tensor.var.TensorVariable
        :param ist: Source variable (output of the network)

        :type soll: theano.tensor.var.TensorVariable or int
        :param soll: Target variable (or a dummy variable). Set to 0 for autoencoders, which would use the model
                     (self.x) input as the target. The default None results in self.yt being considered the training
                     target.

        :type params: list
        :param params: List of parameters to compute gradients.

        :type method: str or function
        :param method: Method to use while computing loss. If callable, it must take in: at least (ist, soll) as inputs.

        :type isautoencoder: bool
        :param isautoencoder: Whether the model being trained is an autoencoder

        :type regularizer: str or function
        :param regularizer: Regularizers to use while computing cost. If callable, it must take in at least (params) as
                            input.

        :type vectorize: bool
        :param vectorize: Whether to vectorize the loss function

        :type kwargs: dict
        :param kwargs: Extra arguments for the cost method and/or regularizers (if any).
                       Recommended: {'loss:kwargname1': kwargvalue1, 'loss:kwargname2': kwargvalue2,
                                     'regularizer:kwargvname1': kwargvalue1, ...}
        """

        # Determine whether the model should be an autoencoder
        isautoencoder = isautoencoder or (self.isautoencoder if self.isautoencoder is not None else False)

        # Determine whether the model could be an autoencoder
        if (soll is not None and ((soll == 0) or isautoencoder)) or (((soll is None) or (soll == 0)) and isautoencoder):
            # Model is an autoencoder
            self.isautoencoder = isautoencoder = True
            soll = self.xr
        else:
            # Model is not an autoencoder.
            self.isautoencoder = isautoencoder = False
            # Assign soll to layer target if set to none.
            if soll is None:
                soll = self.yt

        if ist is None:
            if not isautoencoder:
                # Check if the model is fed forward, i.e. if theano graphs are linked. If the user calls this method
                # before calling feedforward(), he/she obviously has no clue about how Antipasti works - so call the
                # feedforward method with the default arguments now.
                if None in [y.owner for y in pyk.obj2list(self.y)]:
                    self.feedforward()
                ist = self.y
            else:
                # Feedforward encoder and/or decoder if the user didn't.
                if None in [y.owner for y in pyk.obj2list(self.xr)]:
                    if None in [y.owner for y in pyk.obj2list(self.y)]:
                        self.feedforward()
                    self.decoderfeedforward()
                ist = self.xr

        # Parse parameters (which should be instantiated by now)
        if params is None:
            params = self.params

        # Parse kwargs
        losskwargs, regkwargs = netutils.splitkwargs(kwargs, 'loss', 'regularizer')

        # Compute loss
        # This allows method to be a function name (string) from the netrain (nt) file.
        try:
            if isinstance(method, str):
                # And this is why I love Python:
                # Smartfunc converts the function to a "smart" function, which ignores the keyword arguments it
                # doesn't need. Vectorizefunc vectorizes the function (i.e. makes it work on lists) if it's required to
                L = (netutils.vectorizefunc if vectorize else lambda fun: fun)(netutils.smartfunc(getattr(nt, method)))\
                    (ist=ist, soll=soll, isautoencoder=isautoencoder, **losskwargs)
            elif callable(method):
                L = (netutils.vectorizefunc if vectorize else lambda fun: fun)(netutils.smartfunc(method))\
                    (ist=ist, soll=soll, isautoencoder=isautoencoder, **losskwargs)
            else:
                raise NotImplementedError("Cost method evaluation failed.")

            # Reduce loss to a sum
            L = sum(pyk.obj2list(L))

        except AttributeError as e:
            print("Original Error: {}".format(e.message))
            raise NotImplementedError("Loss method {} not implemented.".format(method))

        # Compute regularizers
        try:
            if isinstance(regularizer, str):
                R = netutils.smartfunc(getattr(nt, regularizer))(params=params, **regkwargs)
            elif callable(regularizer):
                R = netutils.smartfunc(regularizer)(params=params, **regkwargs)
            else:
                raise NotImplementedError("Regularization method evaluation failed.")
        except AttributeError as e:
            print("Original Error: {}".format(e.message))
            raise NotImplementedError("Regularizer method {} not implemented.".format(method))

        # Compute cost
        C = L + R

        # Compute gradient
        pdC = T.grad(C, wrt=params, disconnected_inputs='warn')

        if nanguard:
            # Kill gradients if cost is nan
            dC = [th.ifelse.ifelse(T.isnan(C) or T.any(T.isnan(dparam)), T.zeros_like(dparam), dparam) for dparam in pdC]
        else:
            # Don't kill gradients
            dC = pdC

        # Set class attributes and return
        self.C, self.dC, self.L = C, dC, L
        return C, dC, L

    # Method to get updates
    def getupdates(self, cost=None, gradients=None, method='sgd', **kwargs):
        """
        :type cost: theano.tensor.var.TensorVariable
        :param cost: Cost scalar

        :type gradients: list
        :param gradients: List of gradients w.r.t. the corresponding element in the list of parameters

        :type method: str or callable
        :param method: Method for weight update. If callable, should take (params, cost, gradient), in that order.

        :type kwargs: dict
        :param kwargs: Extra arguments for method (if any)
        """

        # Parse cost and gradient
        if cost is None:
            cost = self.C

        if gradients is None:
            gradients = self.dC

        # Make sure there are no ghost variables lurking in the parameter list
        assert not any([isinstance(param, netutils.ghostvar) for param in self.params]), \
            "Uninstantiated ghost variables found in the parameter list. Run feedforward() or cost() method first."

        if method in ['sgd', 'stochastic gradient descent']:
            self.updates = nt.sgd(self.params, cost=cost, gradients=gradients,
                                  learningrate=kwargs["learningrate"] if "learningrate" in kwargs.keys() else None)
        else:
            # This allows method to be a function name string from the netrain py file.
            try:
                if isinstance(method, str):
                    self.updates = netutils.smartfunc(getattr(nt, method))(params=self.params, cost=cost,
                                                                           gradients=gradients, **kwargs)
                elif callable(method):
                    self.updates = netutils.smartfunc(method)(self.params, cost=cost, gradients=gradients, **kwargs)
                else:
                    raise NotImplementedError("Update method evaluation failed.")

            except AttributeError:
                raise NotImplementedError("Update method {} not implemented.".format(method))

        # Append update requests
        self.updates += self.updaterequests

        # Return
        return self.updates

    # Method to compile model functions
    def compile(self, what='trainer', isautoencoder=None, fetchgrads=True, extrarguments={}, compilekwargs=None):
        """
        :type what: str
        :param what: Compile what? Possible keys: "trainer", "inference", "error"

        :type isautoencoder: bool
        :param isautoencoder: Whether to compile an autoencoder.

        :type fetchgrads: bool
        :param fetchgrads: Whether to fetch gradients. Can be laggy for a lot of parameters in the gradient.

        :type extrarguments: dict
        :param extrarguments: Extra arguments to the compiled function.
                       - Keys must be theano variables, and
                       - Values must be generators, i.e. contain a next() method OR a callable(costlog, losslog)
                       Optional: restartgenerator() method to unwind generators when a new epoch begins.
        """

        if isautoencoder is None:
            isautoencoder = self.isautoencoder

        if compilekwargs is None:
            compilekwargs = {}

        if what is "trainer" or what is "all":
            # Compile classifier trainer
            if not isautoencoder:
                # Compile
                classifiertrainer = th.function(inputs=pyk.obj2list(self.x) + pyk.obj2list(self.yt) +
                                                       extrarguments.keys(),
                                                outputs=[self.C, self.L, self.E] + (self.dC if fetchgrads else []),
                                                updates=self.updates,
                                                allow_input_downcast=True,
                                                on_unused_input='warn', **compilekwargs)
                self.classifiertrainer = classifiertrainer
                return classifiertrainer
            else:
                # Compile autoencoder trainer
                autoencodertrainer = th.function(inputs=pyk.obj2list(self.x) + extrarguments.keys(),
                                                 outputs=[self.C, self.L, self.E] + (self.dC if fetchgrads else []),
                                                 updates=self.updates,
                                                 allow_input_downcast=True,
                                                 on_unused_input='warn', **compilekwargs)
                self.autoencodertrainer = autoencodertrainer
                return autoencodertrainer

        if what is "inference" or what is "all":
            if not isautoencoder:
                classifier = th.function(inputs=pyk.obj2list(self.x),
                                         outputs=self.y,
                                         allow_input_downcast=True,
                                         on_unused_input='warn')
                self.classifier = classifier
                return classifier
            else:
                autoencoder = th.function(inputs=pyk.obj2list(self.x),
                                          outputs=self.xr,
                                          allow_input_downcast=True,
                                          on_unused_input='warn')
                self.autoencoder = autoencoder
                return autoencoder

        if what is "error" or what is "all":
            if not isautoencoder:
                classifiererror = th.function(inputs=pyk.obj2list(self.x) + pyk.obj2list(self.yt),
                                              outputs=self.E,
                                              allow_input_downcast=True,
                                              on_unused_input='warn')
                self.classifiererror = classifiererror
                return classifiererror
            else:
                reconstructionerror = th.function(inputs=pyk.obj2list(self.x),
                                                  outputs=self.E,
                                                  allow_input_downcast=True,
                                                  on_unused_input='warn')
                self.reconstructionerror = reconstructionerror
                return reconstructionerror

    # Method to train the model
    def fit(self, trX, trY=None, numepochs=100, maxiter=np.inf, verbosity=0, progressbarunit=1,
            vaX=None, vaY=None, validateevery=None, recompile=False, nanguard=True, extrarguments={}, circuitX=None,
            circuitY=None, trainmonitors=None, validatemonitors=None, backupparams=True, backupbestparams=True):
        """
        :type trX: generator
        :param trX: Generator for the training X data (i.e. images). Must have a next() and restartgenerator() method.

        :type trY: generator
        :param trY: Generator for the training Y data (i.e. labels). Must have a next() and restartgenerator() method.
                    Omit to train an autoencoder, or set to -1 if generator trX.next() returns both X and Y batches

        :type numepochs: int
        :param numepochs: int

        :type verbosity: int
        :param verbosity: Verbosity. 0 for silent execution, 4 for full verbosity.

        :type progressbarunit: int
        :param progressbarunit: Print training progress every `progressbarunit` iterations.

        :type vaX: generator
        :param vaX: Generator for the validation X data (i.e. images). Must have a next() method.

        :type vaY: generator
        :param vaY: Generator for the validation X data (i.e. images). Must have a next() method.

        :type validateevery: int
        :param validateevery: Validate every validateevery iteration.

        :type recompile: bool
        :param recompile: Whether to recompile trainer functions even if a precompiled version is available in the cache

        :type nanguard: bool
        :param nanguard: Breaks training loop if cost or loss is found to be NaN.

        :type extrarguments: dict
        :param extrarguments: Extra arguments to the compiled function.
                       - Keys must be theano variables, and
                       - Values must be generators, i.e. contain a next() method OR a callable(costlog, losslog)
                       Optional: restartgenerator() method to unwind generators when a new epoch begins.

        :type trainmonitors: list of callables
        :param trainmonitors: Training Monitors (see netools for a few pre-implemented monitors)

        :type validatemonitors: list of callables
        :param validatemonitors: Validation Monitors (see netools for a few pre-implemented monitors)

        :type backupparams: bool or int
        :param backupparams: Whether to backup parameters every time a better set of parameters are found (if bool).
                             If int, save parameters every backupparams iterations.

        :type backupbestparams: bool
        :param backupbestparams: Whether to backup best set of parameters
        """

        # This function will:
        #   Compile a trainer function if not precompiled
        #   Run trainer batch-wise

        # Check if inputs are correct
        assert trY is None or trY is -1 or hasattr(trY, "next"), "trY must have a next method or equal -1 if Y batch" \
                                                                 " is returned by the next() method of trX."
        assert hasattr(trX, "next"), "trX must have a next method."
        assert vaY is None or vaY is -1 or hasattr(vaY, "next"), "vaY must have a next method or equal -1 if Y batch" \
                                                                 " is returned by the next() method of vaX."
        assert hasattr(vaX, "next") or vaX is None, "vaX must have a next method."

        # Confirm that the model is an autoencoder if trY is not given
        assert trY is not None or self.isautoencoder, "Training targets are required for non-autoencoding networks."

        # Confirm if all theano graphs linked.
        assert None not in [self.C.owner, self.L.owner], "Select cost function with the cost() method before fitting."
        assert None not in [y.owner for y in pyk.obj2list(self.y)] if not self.isautoencoder else \
            None not in [xr.owner for xr in pyk.obj2list(self.xr)] is not None, \
            "Theano graph is not built correctly. Consider calling feedforward() or decoderfeedforward() followed " \
            "by cost()."
        # Check if monitors are valid
        assert trainmonitors is None or all([callable(monitor) for monitor in trainmonitors]), \
            "Training Monitors must be callable functions."
        assert validatemonitors is None or all([callable(monitor) for monitor in validatemonitors]), \
            "Validation Monitors must be callable functions."

        if verbosity >= 2:
            print("Tests passed. Training {} (ID: {})...".format("Autoencoder" if self.isautoencoder else "Classifier",
                                                                 id(self)))

        # Parse Monitors
        if trainmonitors is None:
            trainmonitors = [ntl.batchmonitor, ntl.costmonitor, ntl.lossmonitor, ntl.trEmonitor, ntl.gradnormmonitor,
                             ntl.updatenormmonitor] \
                if verbosity >= 4 else \
                [ntl.batchmonitor, ntl.costmonitor, ntl.lossmonitor] \
                if verbosity >= 3 else \
                [ntl.batchmonitor] \
                if verbosity >= 2 else []
        if validatemonitors is None:
            validatemonitors = [ntl.batchmonitor, ntl.costmonitor, ntl.lossmonitor, ntl.vaEmonitor] \
                if verbosity >= 4 else \
                [ntl.batchmonitor, ntl.costmonitor, ntl.lossmonitor] \
                if verbosity >= 3 else \
                [ntl.batchmonitor] \
                if verbosity >= 2 else []

        # Link error variable if not done already
        if self.E.owner is None:
            self.error()

        # Compile trainers
        if (self.classifiertrainer is None and not self.isautoencoder) or (self.autoencodertrainer is None
                                                                           and self.isautoencoder) or recompile:
            if verbosity >= 1:
                print("Compiling Trainer...")

            self.compile(what="trainer", extrarguments=extrarguments)

        # Set up validation
        validate = not(validateevery is None or validateevery == 0 or (vaY is None and not self.isautoencoder)
                       or vaX is None)

        # UI
        if verbosity >= 2:
            print("Validation Status: {}...".format("Active" if validate else "Inactive"))

        # Check if validation functions compiled
        if self.classifiererror is None and not self.isautoencoder and validate:
            if verbosity >= 1:
                print("Compiling Validation Function...")
            
            self.compile(what="error", isautoencoder=False, extrarguments=extrarguments)

        if self.reconstructionerror is None and self.isautoencoder and validate:
            if verbosity >= 1:
                print("Compiling Validation Function...")
            # Compile
            self.compile(what="error", isautoencoder=True, extrarguments=extrarguments)

        # Loop variables
        numiter = 0
        costlog = []
        losslog = []
        errorlog = []
        validationlog = []
        skipvalidation = False
        skipepoch = False
        bestparams = []
        # Loop over epochs
        for epoch in range(numepochs):
            # Break if maximum number if iterations exceeded
            if numiter >= maxiter:
                break

            if skipepoch:
                continue

            # Print
            if verbosity >= 1:
                print("Training Epoch: {} of {}".format(epoch, numepochs))

            # Loop variables:
            batchnum = 0
            # Loop over batches
            while True:
                # Break if maximum number of iterations exceeded
                if numiter >= maxiter:
                    # Print
                    if verbosity >= 1:
                        print("Maximum number of iterations exhausted, aborting...")
                    break

                # Break if any of the generators exhausted
                try:
                    # Check if trX returns both X and Y batches simultaneously.
                    if trY is -1:
                        # Fetch everything the generator has to return
                        batch = trX.next()
                        # Get X and Y batches
                        if circuitX is None and circuitY is None:
                            batchX = batch[0:self.numinp]
                            batchY = batch[self.numinp:(self.numinp + self.numout)]
                        else:
                            try:
                                batchX = batch[0:self.numinp][circuitX]
                                batchY = batch[self.numinp:(self.numinp + self.numout)][circuitY]
                            except:
                                batchX = batch[0:self.numinp]
                                batchY = batch[self.numinp:(self.numinp + self.numout)]

                        # Get extra arguments if any
                        if len(batch) > (self.numinp + self.numout):
                            # Fetch remaining arguments as possible extra arguments
                            batchEA = batch[(self.numinp + self.numout):]
                        else:
                            batchEA = []
                    else:
                        # Fetch X
                        batchX = trX.next()
                        # Fetch Y if not autoencoder
                        batchY = trY.next() if not self.isautoencoder else None
                        # FIXME: Pretend no extra arguments are provided
                        batchEA = []

                    # Fetch extra inputs
                    extrargs = []
                    eacursor = 0
                    for arg in extrarguments.values():
                        # Try fetching from main generators
                        if arg == -1:
                            extrargs.append(batchEA[eacursor])
                            eacursor += 1
                            continue

                        # Try fetching from the next method
                        elif hasattr(arg, "next"):
                            extrargs.append(arg.next())
                            continue

                        # Try calling the object
                        elif callable(arg):
                            extrargs.append(arg(costlog, losslog))
                            continue

                        else:
                            raise NotImplementedError("Extra arguments must be provided with the main generators, "
                                                      "or have a next() method, or be a callable.")

                    # Make sure all extra arguments are in place
                    assert all([arg is not None for arg in extrargs]), "Extra arguments must have a " \
                                                                       "next() method or be callable."

                except StopIteration:
                    if verbosity >= 2:
                        print("Generator(s) exhausted...")
                    break

                # Build function input. batchX and batchY might be tuples or lists for all we know
                funin = pyk.obj2list(batchX) + (pyk.obj2list(batchY) if not self.isautoencoder else []) + extrargs

                # Run trainer
                try:
                    res = self.classifiertrainer(*funin) if not self.isautoencoder else \
                        self.autoencodertrainer(*funin)
                    C, L, E = res[0:3]
                    dC = res[3:]
                except Exception as e:
                    # Save network parameters
                    self.save(nameflags='-mayday')
                    # UI
                    if verbosity >= 1:
                        print("Exception raised, backing up parameters as {}...".format(self.lastsavelocation))
                    # Raise original exception
                    raise e

                # NaNGuard
                if nanguard and (np.isnan(C) or np.isnan(L)):
                    if verbosity >= 2:
                        print("NaNs detected, aborting training...")
                    skipepoch = True
                    break

                # Provision to add arbitrary monitors
                # How-to: Use a smart function with the following possible kwargs:
                #   - batchnum
                #   - cost
                #   - loss
                #   - trE (trainingerror)
                #   - vaE (validationerror)
                #   - dC (gradient)
                #   - params
                # Pro-tip: Functions can have persistent attributes!

                if numiter % progressbarunit == 0:
                    # Try to fetch numerical params
                    if hasattr(self, "getparams"):
                        params = self.getparams()[0]
                    else:
                        params = self.params

                    # Build a dictionary of keyword arguments to send to the monitors
                    iterstat = {"batchnum": batchnum, "cost": C, "loss": L, "trE": E, "dC": dC, "params": params}

                    # Get outputs from training monitors
                    monitorout = [netutils.smartfunc(monitor, ignorekwargssilently=True)(**iterstat)
                                  for monitor in trainmonitors]
                    # Get rid of Nones if any
                    monitorout = [out for out in monitorout if out is not None]
                    # Print Monitor
                    print("".join(monitorout))

                # Log
                costlog.append(C)
                losslog.append(L)
                errorlog.append(E)

                # Check if any validation to be done in this iteration
                if validate and numiter % validateevery == 0 and numiter != 0:
                    try:
                        if vaY is -1:
                            vbatch = vaX.next()
                            vbatchX = vbatch[0:self.numinp]
                            vbatchY = vbatch[self.numinp:(self.numinp + self.numout)]
                        else:
                            vbatchX = vaX.next()
                            vbatchY = vaY.next() if not self.isautoencoder else None
                        skipvalidation = False
                    except StopIteration:
                        # Restart generators and try anew
                        if hasattr(vaX, "restartgenerator"):
                            vaX.restartgenerator()
                        if not self.isautoencoder and hasattr(vaY, "restartgenerator"):
                            vaY.restartgenerator()
                        # Try to Fetch
                        try:
                            if vaY is -1:
                                vbatch = vaX.next()
                                vbatchX = vbatch[0:self.numinp]
                                vbatchY = vbatch[self.numinp:(self.numinp + self.numout)]
                            else:
                                vbatchX = vaX.next()
                                vbatchY = vaY.next() if not self.isautoencoder else None
                            skipvalidation = False
                        except:
                            warn("Failed to fetch validation data...")
                            skipvalidation = True

                    if not skipvalidation:
                        vfunin = pyk.obj2list(vbatchX) + (pyk.obj2list(vbatchY) if not self.isautoencoder else [])
                        vaE = self.classifiererror(*vfunin) if not self.isautoencoder \
                            else self.reconstructionerror(vbatchX)
                    else:
                        vaE = None

                    if vaE is not None and verbosity >= 2:
                        print("Validation Error: {}...".format(vaE))

                    validationlog.append(vaE)

                    # TODO: Validation Monitors

                if backupparams:
                    # Back up params every backupparams iteration if backupparams is an integer
                    if isinstance(backupparams, int):
                        if numiter % backupparams == 0 and numiter is not 0:
                            if verbosity >= 3:
                                print("Backing up parameters (routine)...")
                            self.save(nameflags='-routine')

                    # Log parameters with the best validation error (if available) or with the best training error
                    if validate and None not in validationlog and len(validationlog) > 1:
                        if validationlog[-1] < min(validationlog[:-1]) and numiter % validateevery == 0 and numiter != 0:
                            bestparams = self.params
                            # Back up parameters if requested
                            if backupbestparams:
                                if verbosity >= 3:
                                    print("Backing up best set of parameters...")
                                self.save(nameflags='-best')
                    elif len(errorlog) > 1 and not validate:
                        if errorlog[-1] < min(errorlog[:-1]):
                            bestparams = self.params
                            # Back up parameters if requested
                            if backupbestparams:
                                if verbosity >= 3:
                                    print("Backing up best set of parameters...")
                                self.save(nameflags='-best')

                # Increment iteration counter
                numiter += 1
                # Increment batch count
                batchnum += 1

            # Restart generators
            for arg in [trX, trY] + extrarguments.values():
                if hasattr(arg, "restartgenerator"):
                    arg.restartgenerator()

        # Training done, apply best parameters and call it a day
        if not bestparams == []:
            self.applyparams(params=bestparams)

        # Return
        return costlog, losslog, errorlog, validationlog

        # Save parameters and model
    def save(self, where=None, nameflags='', overwrite=True):
        if where is None:
            if self.savedir is None:
                # If no save directory is provided, use ../Backups
                if not os.path.isdir("../Backups"):
                    # If ../Backups doesn't exist, create it
                    os.mkdir("../Backups")
                savedir = "../Backups"
            else:
                # Use provided savedir
                savedir = self.savedir
            where = os.path.join(savedir,
                                 "ltp-" + datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d--%H-%M-%S") +
                                 nameflags + ".save")

        # Delete previous save with same nameflag
        if overwrite:
            try:
                # List and sort files in directories
                filelist = sorted(os.listdir(os.path.split(where)[0]))
                # Get last element in the list containing the provided nameflag
                try:
                    # Find file to delete
                    deletefile = next(x for x in reversed(filelist)
                                      if bool(re.match("^ltp-\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}" +
                                                       nameflags + "\.save$", x))) \
                        if nameflags is not '' else None
                except StopIteration:
                    deletefile = None

                # Delete deletefile if there's a deletefile to be deleted (tehe)
                if deletefile is not None:
                    os.remove(os.path.join(os.path.split(where)[0], deletefile))

            except Exception:
                warn("Could not delete previous save.")

        # Log save location
        self.lastsavelocation = where

        # Save parameters and cparameters
        f = file(where, mode='wb')
        for obj in [self.params, self.cparams]:
            pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()

    def load(self, fromwhere=None):
        # When not specified, load from the last saved location
        if fromwhere is None:
            fromwhere = self.lastsavelocation

        # Load params and cparams
        f = file(fromwhere, mode='rb')
        loadlist = []
        for _ in range(2):
            try:
                loadlist.append(pkl.load(f))
            except EOFError:
                warn("Failed to load {} from {}...".format({0: "params", 1: "cparams"}[_], fromwhere))
                loadlist.append(None)
        f.close()

        # Apply parameters and cparameters
        self.applyparams(params=loadlist[0], cparams=loadlist[1])

    def pickle(self, where=None):
        if where is None:
            # Make a sibling directory backup if it doesn't exist already
            if not os.path.isdir("../Backups"):
                os.mkdir("../Backups")
            where = os.path.join("../Backups",
                                 "model_{}-".format(str(id(self))) +
                                 datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d--%H-%M-%S") + ".save")

        # Pickle self FIXME this does not work because there's a callable which must be pickled. try pathos dill.
        f = file(where, mode='wb')
        pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()


class layertrain(model):
    """ Class to handle a stack of layers (Feedforward network) """

    # Constructor
    def __init__(self, train, active=None, inpshape=None):
        """
        :type train: list of nk.layer
        :param train: A train (in choo choo sense) of layers

        :type active: slice
        :param active: Slice of the train which is active. The remaining coaches do not get updated by the
                       optimizer.

        :type inpshape: tuple or list
        :param inpshape: Shape of the expected input
        """

        # Init superclass
        # Initialize super class
        super(layertrain, self).__init__()

        # Check inputs and convert train to list if it isn't one
        if not isinstance(train, list):
            train = [train]

        # Totally meta
        self._train = train
        self._activetrain = None
        self._params = None
        self._cparams = None
        self._inpshape = None
        self.outdim = None
        self.updaterequests = []
        self.active = active
        self.recurrencenumber = 0.5

        # Compatibility with layertrainyard
        self.numinp = 1
        self.numout = 1

        # Compute the number of input feature maps, and the input dimension
        self.fmapsin = None
        self.fmapsout = None
        self.inpdim = self.train[0].inpdim

        for layer in self.train:
            if isinstance(layer, nk.convlayer):
                self.fmapsin = layer.fmapsin
                break

        for layer in self.train[::-1]:
            if isinstance(layer, nk.convlayer):
                self.fmapsout = layer.fmapsout
                break

        # Parse active: this is where train, params and cparams gets parsed and initialized. This has to be done before
        # inpshape is set.
        if active is None:
            self.activetrain = self.train
        else:
            assert isinstance(active, slice), "Active must be a slice object."
            assert active.stop <= len(train), "Slice stop exceeds train length."
            self.activetrain = self.train[active]

        # Parse input shape. Set it to that of the first layer in train if none given
        if inpshape is None:
            self.inpshape = [None, ] * self.inpdim if self.train[0].inpshape is None else self.train[0].inpshape
        else:
            assert len(inpshape) == self.inpdim, "Length of inshape must equal the number of input dimensions."
            self.inpshape = inpshape

        # Meta
        self.allparams = [param for layer in self.train for param in layer.params]
        self.allcparams = [cparam for layer in self.train for cparam in layer.cparams]

        # Update requests
        self.allupdaterequests = [request for layer in self.train for request in layer.updaterequests]

        # Containers for model input, output and reconstruction
        self.x = T.tensor('floatX', [False, ] * self.inpdim, 'model-x:'+str(id(self)))
        self.y = T.tensor('floatX', [False, ] * self.outdim, 'model-y:'+str(id(self)))
        self.xr = T.tensor('floatX', [False, ] * self.inpdim, 'model-xr:'+str(id(self)))

        # Container for target
        self.yt = T.tensor('floatX', [False, ] * self.outdim, 'model-yt:'+str(id(self)))

        # Trainer
        self.trainer = None

        # IO
        # Last save location
        self.lastsavelocation = None

    # Activetrain is set by active, but add a method to set it otherwise (this steals some usability from active,
    # but could come in handy nonetheless)
    @property
    def activetrain(self):
        return self._activetrain

    @activetrain.setter
    def activetrain(self, value):
        # Check if value a layertrain, in which case set value to its train
        value = value.train if isinstance(value, layertrain) else value
        # Convert value to list if it isn't one
        value = value if isinstance(value, list) else [value]

        # Keep uniques
        value = pyk.unique(value)

        # Check if value is a list of objects in train
        assert all([val in self.train for val in value]), "Activetrain must be assigned a list of objects already in " \
                                                          "train. Correct: activetrain = train[...]. Incorrect: " \
                                                          "activetrain = [convlayer(...), ...]"

        # Assign activetrain
        self._activetrain = value

        # Rebuild parameter list
        self.rebuildparamlist(activetrain=value)
        # Rebuild update request list
        self.rebuildupdaterequestlist(activetrain=value)

    @property
    def train(self):
        return self._train

    @model.inpshape.setter
    def inpshape(self, value):
        self._inpshape = value
        self.outshape = self.inferoutshape(inpshape=value)
        self.outdim = len(self.outshape)
        # Params and cparams within some layers might have changed. Rebuild param and cparam lists to mirror these
        # changes.
        self.rebuildparamlist()

    # Method to instantiate all ghost variables in params
    def instantiate(self):
        # Only ghost variables in the required layers are to be instantiated, i.e. those in self._params
        # (and not in self.allparams). This is because it might be the case that the user specifies the input shape of a
        # given layer in train but not of the train itself. In this case, one might not know the inpshape of all layers
        # in train.
        for layer in self.activetrain:
            layer.instantiate()
        # Instantiateparams() method changes the params attribute of the corresponding layer. These changes are to be
        # mirrored in current paramlist.
        self.rebuildparamlist()
        pass

    # Method to rebuild parameter list
    def rebuildparamlist(self, activetrain=None, train=None):
        # Parse active train
        if activetrain is None:
            activetrain = self.activetrain
        # Parse train
        if train is None:
            train = self.train

        # Assign _params (i.e. active params or params in active train)
        self._params = [param for layer in activetrain for param in layer.params]
        self._cparams = [cparam for layer in activetrain for cparam in layer.cparams]

        # Assign all params (i.e. all parameters in model)
        self.allparams = [param for layer in train for param in layer.params]
        self.allcparams = [cparam for layer in train for cparam in layer.cparams]

    def rebuildupdaterequestlist(self, activetrain=None, train=None):
        # Parse active train and train
        if activetrain is None:
            activetrain = self.activetrain
        if train is None:
            train = self.train

        # Assign updaterequests
        self.updaterequests = [request for layer in activetrain for request in layer.updaterequests]
        # Assign allupdaterequests
        self.allupdaterequests = [request for layer in train for request in layer.updaterequests]

    # Method to refresh the state of the layertrain object (i.e. rebuild parameter lists and update requests)
    def refresh(self):
        # Rebuild parameter list
        self.rebuildparamlist()
        # Rebuild update request list
        self.rebuildupdaterequestlist()

    # TODO: Check train for errors or breached duck contracts
    def checktrain(self, train):
        pass

    # Add layers to train
    def __add__(self, other):
        # Make sure the number of inputs/outputs check out
        assert self.numout == other.numinp, "Cannot chain a component with {} output(s) " \
                                            "with one with {} input(s)".format(self.numout, other.numinp)
        if isinstance(other, layertrain):
            return layertrain(self.train + other.train)
        elif isinstance(other, nk.layer):
            # Layer may have one or multiple inputs/outputs
            if other.numout > 1:
                return layertrainyard([self, other])
            else:
                return layertrain(self.train + [other])
        elif isinstance(other, layertrainyard):
            return layertrainyard([self] + other.trainyard)
        else:
            raise TypeError('Second summand of invalid type.')

    # Multiply to layertrainyard
    def __mul__(self, other):
        if isinstance(other, nk.layer):
            return layertrainyard([[self, other]])
        elif isinstance(other, layertrain):
            return layertrainyard([[self, other]])
        elif isinstance(other, layertrainyard):
            return layertrainyard([[self, other]])
        else:
            raise TypeError('Second summand of invalid type.')

    # Index layers to access subnetworks.
    def __getitem__(self, item):
        # Item could either be a number or a slice, or even a string with the layer-id
        if isinstance(item, (int, float)):
            return self.train[item]
        elif isinstance(item, slice):
            return layertrain(self.train[item])
        elif isinstance(item, str):
            # Parse layer from string
            try:
                idx = [str(id(layer)) for layer in self.train].index(item)
                return layertrain(self.train[idx:idx + 1])
            except ValueError as e:
                print("Original exception message: {}".format(e.args[0]))
                raise ValueError("Layer not found in layertrain...")

    # Train length
    def __len__(self):
        return len(self.train)

    # Iterate over network
    def __iter__(self):
        return iter(self.train)

    # Method to set a sublist
    def __setitem__(self, key, value):
        if isinstance(value, list):
            pass
        elif isinstance(value, layertrain):
            value = value.train
        else:
            raise NotImplementedError("Assigned value must be a list of layers or a layertrain!")

        train = self.train
        train.__setitem__(key, value)
        return layertrain(train)

    # Call method for running the forward pass for a given batch (numerical). Inspired by Torch's forward method
    def __call__(self, inp):
        # Check if the network is fed forward
        assert self.y.owner is not None, "Run feedforward() method first!"
        # Evaluate and return
        return self.y.eval({self.x: inp.astype(th.config.floatX)})

    # Define feed forward method
    def feedforward(self, inp=None):

        # Instantiate ghost parameters if there are any
        self.instantiate()

        # Parse inp. Set x of first layer of train and that of the model (superclass) to inp when yes.
        if inp is not None:
            self.train[0].x = inp
            self.x = inp
        else:
            # Assume model input has been set.
            self.train[0].x = self.x

        # Return input if train empty
        if not self.train:
            self.y = inp
            return inp

        # Feed forward through hidden layers
        for layernum in range(len(self) - 1):
            self.train[layernum + 1].x = self.train[layernum].feedforward()

        # Feed forward through the last (output) layer and assign to y of the model
        self.y = self.train[-1].feedforward()

        # There might be new update requests to fetch
        self.rebuildupdaterequestlist()

        # return
        return self.y

    # Define decoder feed forward method
    def decoderfeedforward(self, inp=None):

        # Reverse train to decode
        revtrain = copy.copy(self.train)
        revtrain.reverse()

        # Check if train empty, return inp
        if not revtrain:
            return inp

        # Check if all decoders active.
        if not all([layer.decoderactive for layer in revtrain]):
            warn('Not all decoders are active. Ignore this warning if this is intentional.')

        # Parse and set y of model to input.
        if inp:
            revtrain[0].y = inp
            self.y = inp
        else:
            revtrain[0].y = self.y

        # Decode away
        for layernum in range(len(revtrain) - 1):
            revtrain[layernum + 1].y = revtrain[layernum].decoderfeedforward()

        # Decode the last layer and assign to model output
        self.xr = revtrain[-1].decoderfeedforward()

        # return
        return self.xr

    # Step method is an alias for feedforward (required for use in recurrent chains)
    def step(self, inp):
        return self.feedforward(inp)

    # Define parameter rolling scheme
    def rollparameters(self, params=None):
        # Parse input
        if not params:
            params = self.params

        # Check if given params numpy or Theano
        if all([isinstance(param, np.ndarray) for param in params]):
            # Flatten, concatenate and return in numpy
            return np.concatenate([param.flatten() for param in params])
        else:
            # Flatten, concatenate and return in theano
            return T.concatenate(map(T.flatten, params))

    # Define parameter unrolling scheme
    def unrollparameters(self, paramvec):
        # Validate shape
        assert isinstance(paramvec, np.ndarray) and np.size(paramvec.shape) == 1, \
            'Parameter vector must be a 1D numpy ndarray.'

        # define cursors on the parameter vector
        cursorstart = cursorstop = 0

        # initialize list to hold parameters
        params = []

        # Loop over parameters in network
        for param in self.params:
            # Obtain parameter shape of param (assuming param is a theano shared variable)
            parshape = param.shape.eval()
            # Obtain the number of elements in param
            parnumel = np.prod(parshape)

            # Set cursorstop
            cursorstop = cursorstart + parnumel - 1
            # Fetch from vector and reshape to parameter shape before appending to params
            params.append(paramvec[cursorstart:cursorstop].reshape(parshape))

            # Update cursorstart
            cursorstart = cursorstop + 1

        # return
        return params

    # Apply parameters to network
    def applyparams(self, params=None, cparams=None):

        if params:
            # Check if params nd-vector (unrolled) or list of params
            if isinstance(params, np.ndarray):
                params = self.unrollparameters(params)
            elif not isinstance(params, list):
                raise TypeError('The argument params must be a list of parameters or a 1D numpy ndarray.')

        if cparams:
            # Check if cparams nd-vector (unrolled) or list of params
            if isinstance(cparams, np.ndarray):
                params = self.unrollparameters(cparams)
            elif not isinstance(cparams, list):
                raise TypeError('The argument cparams must be a list of parameters or a 1D numpy ndarray.')

        # Generate a list where the k-th element gives the number of weights & biases layer k takes
        lenlist = [len(layer.params) for layer in self.activetrain]

        # Initialize start and stop cursors
        cursorstart = cursorstop = 0

        # Loop over layers and layernums
        for layernum, layer in enumerate(self.activetrain):

            # Skip applying parameters if there are no parameters to apply
            if lenlist[layernum] == 0:
                continue

            # Find where to stop cursor
            cursorstop = cursorstart + lenlist[layernum]

            # Fetch layer (c)parameters and apply to layer
            if params is not None:
                layerparams = params[cursorstart:cursorstop]
            else:
                layerparams = None

            if cparams is not None:
                layercparams = cparams[cursorstart:cursorstop]
            else:
                layercparams = None

            # Apply. Params does not need updating because the numerical values are assigned to theano shared variables,
            # which are pointed at by params and/or cparams
            layer.applyparams(layerparams, layercparams)

            # Find where to start cursor in next iteration
            cursorstart = cursorstop

    # Fetch network parameters
    def getparams(self, allparams=False):
        try:
            numparams = [param.get_value() for param in (self.allparams if allparams else self.params)]
        except AttributeError:
            numparams = [param.eval() for param in (self.allparams if allparams else self.params)]

        try:
            numcparams = [cparam.get_value() for cparam in (self.allcparams if allparams else self.cparams)]
        except AttributeError:
            numcparams = [cparam.eval() for cparam in (self.allcparams if allparams else self.cparams)]

        return numparams, numcparams

    # Infer output shape
    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        if checkinput:
            # The following makes the inferoutshape method raise an error if input shapes are off
            # Track shape changes through the layer train
            shape = inpshape
            for layer in self.train:
                # Set input shape
                layer.inpshape = shape
                shape = layer.outshape
                # Inpshape's setter sets the outshape
            return shape
        else:
            shape = inpshape
            for layer in self.train:
                layer._inpshape = shape
                shape = layer.outshape = layer.inferoutshape(checkinput=False)
            return shape

    # Method to copy a layertrain
    def copy(self):
        return layertrain(self.train, active=self.active, inpshape=self.inpshape)

    def __repr__(self):
        desc = "inp "
        for layer in self.train:
            desc += layer.__repr__()

        desc += " out"
        return desc


# Class to handle networks with multiple streams
class layertrainyard(model):

    def __init__(self, trainyard, inpshape=None):
        """
        :type trainyard: list of layertrainyard or list of layertrain or list of netkit.layer
        :param trainyard: Train yard
        :return:
        """
        # Initialize superclass
        super(layertrainyard, self).__init__()

        # Meta
        self._trainyard = None
        self._activetrainyard = None
        self._params = None

        # Init
        self.trainyard = trainyard

        # Fetch number of inputs
        self.numinp = self.trainyard[0].numinp if not isinstance(self.trainyard[0], list) else \
            sum([train.numinp for train in self.trainyard[0]])
        self.numout = self.trainyard[-1].numout if not isinstance(self.trainyard[-1], list) else \
            sum([train.numout for train in self.trainyard[-1]])

        # Fetch number of input dimensions. First layer might be a (say) list of layer(train) or a splitlayer,
        # there's no way to be sure.
        self.inpdim = self.trainyard[0].inpdim if pyk.smartlen(self.trainyard[0]) == 1 else \
            [coach.inpdim for coach in self.trainyard[0]]
        # outdim is set by shape inference
        self.outdim = None

        # Shape inference
        if inpshape is None:
            if isinstance(self.trainyard[0], list):
                self.inpshape = pyk.delistlistoflists([coach.inpshape for coach in self.trainyard[0]])
            else:
                self.inpshape = self.trainyard[0].inpshape
        else:
            if isinstance(inpshape[0], list):
                assert len(inpshape) == self.numinp, "Layertrainyard has {} inputs, but the provided " \
                                                        "inpshape has {}".format(self.numinp, len(inpshape[0]))
                assert all([len(ishp) == idim for ishp, idim in zip(inpshape, self.inpdim)]), "Input dimension" \
                                                                                                 " mismatch in LTY."
                self.inpshape = inpshape
            else:
                assert self.numinp == 1, "Layertrainyard has {} inputs, but the provided inpshape has 1.".\
                    format(self.numinp)
                self.inpshape = inpshape

        # Containers for inputs may or may-not be lists (for maximum compatibility with model and layertrain)
        self.x = T.tensor('floatX', [False, ]*self.inpdim, name='model-x:'+str(id(self))) \
            if self.numinp == 1 else pyk.chain([pyk.obj2list(coach.x) for coach in self.trainyard[0]]) \
            if pyk.smartlen(self.trainyard[0]) > 1 else self.trainyard[0].x

        self.y = T.tensor('floatX', [False, ]*self.outdim, name='model-y:'+str(id(self))) \
            if self.numout == 1 else pyk.chain([pyk.obj2list(coach.y) for coach in self.trainyard[-1]]) \
            if pyk.smartlen(self.trainyard[-1]) > 1 else self.trainyard[-1].y

        # Container for target, the easy way
        self.yt = pyk.delist([T.tensor('floatX', [False, ] * y.ndim,
                                       name=y.name.replace('y', 'yt') if y.name is not None else None)
                              for y in pyk.obj2list(self.y)])

    @property
    def trainyard(self):
        return self._trainyard

    @trainyard.setter
    def trainyard(self, value):
        # Prune singletons
        value = pyk.removesingletonsublists(value)
        # Loop over elements in value. If they're lists, let them pass; if they're lists of list, replace them with a
        # new layertrainyard. This function will be called recursively and all nested list structures will be cleaned.
        value = [[layertrainyard(elemelem) if isinstance(elemelem, list) else elemelem for elemelem in elem]
                 if isinstance(elem, list) else elem for elem in value]
        # Use trainyard setter to initialize parameters
        self._trainyard = value
        # Rebuild parameter list
        self.rebuildparamlist(trainyard=value)

    @property
    def activetrainyard(self):
        return self._activetrainyard

    @activetrainyard.setter
    def activetrainyard(self, value):
        raise NotImplemented("Use activetrain instead (for now).")

    @model.inpshape.setter
    def inpshape(self, value):
        self._inpshape = value
        self.outshape = self.inferoutshape(inpshape=value)
        self.outdim = pyk.delist([len(oshp) for oshp in pyk.list2listoflists(self.outshape)])
        # Params and cparams might have changed. Rebuild list
        self.rebuildparamlist()

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        # Note that inpshape can be a list of lists
        shape = inpshape
        for train in self.trainyard:
            if isinstance(train, list):
                outshape = []
                # Run shape inference for all coaches in train
                # Convert shape to a list of shapes
                shapelist = pyk.list2listoflists(shape)
                shapelistcursor = 0
                for coach in train:
                    # Fetch the right number of shapes from shapelist
                    coachinpshape = pyk.delist(shapelist[shapelistcursor:shapelistcursor+coach.numinp])
                    shapelistcursor += coach.numinp

                    if checkinput:
                        coach.inpshape = coachinpshape
                        outshape.append(coach.outshape)
                    else:
                        coach._inpshape = coachinpshape
                        coachoutshape = coach.outshape = coach.inferoutshape(coachinpshape, checkinput=False)
                        outshape.append(coachoutshape)

                shape = pyk.delist(pyk.delistlistoflists(pyk.list2listoflists(outshape)))
            else:
                if checkinput:
                    train.inpshape = shape
                    outshape = train.outshape
                else:
                    train._inpshape = shape
                    outshape = train.outshape = train.inferoutshape(checkinput=False)

                shape = pyk.delist(pyk.delistlistoflists(pyk.list2listoflists(outshape)))

        return shape

    def instantiate(self):
        # Instantiate all trains in trainyard (this will try to instantitate layertrains, which will inturn
        # instantiate layers. Duck typing FTW!)
        for train in self.trainyard:
            if isinstance(train, list):
                for coach in train:
                    coach.instantiate()
            else:
                train.instantiate()
        # Mirror changes in params
        self.rebuildparamlist()

    # TODO: Write a decorator that identifies trainyard with train (to have a duck contract with layertrain)
    def rebuildparamlist(self, trainyard=None):
        # Parse
        if trainyard is None:
            trainyard = self.trainyard

        # Build a list of parameters
        self._params = [param for train in trainyard for coach in pyk.obj2list(train) for param in coach.params]
        self._cparams = [cparam for train in trainyard for coach in pyk.obj2list(train) for cparam in coach.cparams]

    def rebuildupdaterequestlist(self, trainyard=None):
        # Parse
        if trainyard is None:
            trainyard = self.trainyard

        # Build list of update requests
        self.updaterequests = [request for train in trainyard for coach in pyk.obj2list(train)
                               for request in coach.updaterequests]

    def refresh(self):
        # Rebuild parameter list
        self.rebuildparamlist()
        # Rebuild update request list
        self.rebuildupdaterequestlist()

    def feedforward(self, inp=None):
        # Parse input
        if inp is not None:
            # Inp is given. Set the x of the first layer in modern
            self.x = inp
            self.trainyard[0].x = inp
        else:
            # Assume model input has been set. Fetch from the list of inputs the correct number of inputs for every
            # coach and set as input
            inplist = pyk.obj2list(self.x)
            inplistcursor = 0
            for coach in pyk.obj2list(self.trainyard[0]):
                # Fetch from the list of inputs
                coachinp = inplist[inplistcursor:inplistcursor+coach.numinp]
                # Increment cursor
                inplistcursor += coach.numinp
                # Delist and set input
                coach.x = pyk.delist(coachinp)

            inp = self.x

        # Complain if trainyard empty
        assert len(self.trainyard) != 0, "Cannot feedforward, trainyard empty."

        # Instantiate
        self.instantiate()

        # Feedforward recursively. Don't take the train-coach analogy too seriously here.
        cache = inp
        for train in self.trainyard:
            if isinstance(train, list):
                # Fetch from the list of inputs the correct number of inputs for every coach and set as input
                inplist = pyk.obj2list(cache)
                inplistcursor = 0
                coachcache = []
                for coach in train:
                    # Fetch coach input
                    coachinp = pyk.delist(inplist[inplistcursor:inplistcursor+coach.numinp])
                    inplistcursor += coach.numinp
                    coachout = coach.feedforward(inp=coachinp)
                    # Store all coach outputs in another cache
                    coachcache.append(coachout)
                cache = coachcache
            else:
                cache = train.feedforward(inp=cache)
            # Flatten any recursive outputs to a linear list
            cache = pyk.delist(list(pyk.flatten(cache)))

        # There might be new update requests to fetch
        self.rebuildupdaterequestlist()

        # Return
        self.y = cache
        return self.y

    def applyparams(self, params=None, cparams=None):
        # Compute the number of parameters per train in trainyard. The structure of lenlist is similar to trainyard
        # itself; if trainyard = [[lt1, lt2], lt3], lenlist = [[3, 0], 4] where 3, 0, and 4 are the number of
        # parameters in lt1, lt2, and lt3 respectively.
        lenlist = [[len(coach.params) for coach in train] if isinstance(train, list) else len(train.params)
                   for train in self.trainyard]

        # Initialize cursors on lenlist
        cursorstart = cursorstop = 0

        # Loop over trains in trainyard
        for trainnum, train in enumerate(self.trainyard):
            # Skip applying parameters if there are no parameters to apply
            if sum(pyk.obj2list(lenlist[trainnum])) == 0:
                continue

            # Update cursorstop
            cursorstop = cursorstart + sum(pyk.obj2list(lenlist[trainnum]))

            if isinstance(train, list):

                # Apply the parameters individually
                subcursorstart = subcursorstop = cursorstart

                for coach, paramlen in zip(train, lenlist[trainnum]):
                    # Update cursor stop
                    subcursorstop = subcursorstart + paramlen
                    if params is not None:
                        # Fetch params
                        coachparams = params[subcursorstart:subcursorstop]
                    else:
                        coachparams = None

                    if cparams is not None:
                        # Fetch cparams
                        coachcparams = cparams[subcursorstart:subcursorstop]
                    else:
                        coachcparams = None

                    # Apply parameters
                    coach.applyparams(params=coachparams, cparams=coachcparams)
                    # Update cursors
                    subcursorstart = subcursorstop

            else:

                # Train is not a list, i.e. params can be applied directly
                if params is not None:
                    trainparams = params[cursorstart:cursorstop]
                else:
                    trainparams = None

                if cparams is not None:
                    traincparams = cparams[cursorstart:cursorstop]
                else:
                    traincparams = None

                # Apply parameters
                train.applyparams(params=trainparams, cparams=traincparams)

            # Update cursor start
            cursorstart = cursorstop

    def __add__(self, other):
        # Make sure the number of inputs/outputs check out
        assert self.numout == other.numinp, "Cannot chain a component with {} output(s) " \
                                            "with one with {} input(s)".format(self.numout, other.numinp)
        # Other could be a layer, layertrain or a layertrainyard
        if isinstance(other, nk.layer):
            return layertrainyard(self.trainyard + [other])
        elif isinstance(other, layertrain):
            return layertrainyard(self.trainyard + [other])
        elif isinstance(other, layertrainyard):
            return layertrainyard(self.trainyard + other.trainyard)
        else:
            raise TypeError("Second summand of invalid type: {}.".format(other.__class__.__name__))

    def __mul__(self, other):
        # Other could be a layer, layertrain or a layertrainyard
        if isinstance(other, nk.layer):
            return layertrainyard([[self, other]])
        elif isinstance(other, layertrain):
            return layertrainyard([[self, other]])
        elif isinstance(other, layertrainyard):
            return layertrainyard([[self, other]])
        else:
            raise TypeError("Second summand of invalid type: {}.".format(other.__class__.__name__))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.trainyard[item]
        elif isinstance(item, slice):
            return layertrainyard(self.trainyard[item])
        else:
            raise NotImplementedError("Index must be an integer or a slice.")
