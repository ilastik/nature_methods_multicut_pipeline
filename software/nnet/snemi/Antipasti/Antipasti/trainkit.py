from matplotlib import pyplot as plt
import bokeh.plotting as bplt

__author__ = "Nasim Rahaman"

__doc__ = """ Module to help monitor and manipulate the training process. """

import yaml
import os
import datetime
import time

import numpy as np
from theano import config

from netutils import sym2num, smartfunc
from netdatautils import pickle, unpickle
import pykit as pyk


class callbacks(object):
    def __init__(self, callbacklist):
        """
        Class to handle a collection of callbacks.

        :type callbacklist: list
        :param callbacklist: List of callbacks
        """
        self.callbacklist = list(callbacklist)

    def __call__(self, *args, **kwargs):
        for cb in self.callbacklist:
            cb(*args, **kwargs)


class callback(object):
    def __init__(self, callevery=1):
        self.callevery = callevery


class logger(callback):
    def __init__(self, logfile, callevery=1):
        """
        A very basic text based logger. The file is open within a context manager, so it's safe against external
        interruptions.

        :type logfile: str
        :param logfile: Path to the logfile.
        """
        super(logger, self).__init__(callevery=callevery)

        self.logfile = logfile

    def log(self, message):
        # Build log message with datetime stamp
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d--%H-%M-%S")
        logmessage = "[{}] - {}\n".format(timestamp, message)
        # Check if log file exists (assign file mode accordingly)
        with open(self.logfile, 'a' if os.path.exists(self.logfile) else 'w') as lf:
            # Log to file
            lf.write(logmessage)

    # For compatibility with file streams
    def write(self, message):
        self.log(message)

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            msg = args[0]
        else:
            msg = kwargs['message']

        self.write(msg)

    def close(self):
        return


class relay(callback):
    def __init__(self, switches, ymlfile, callevery=1):
        """
        Given the path to a YAML file (`ymlfile`) and a dictionary `switches` having the format
        {'name': theano-shared-variable, ...}, this class' read method sets the value of the theano shared variable
        with the one found in the corresponding field of the YAML file.

        :type ymlfile: str
        :param ymlfile: Path to YAML file where the parameters are stored.

        :type switches: dict
        :param switches: Should be {'name': sharedvar, ...}, where sharedvar is a theano shared variable and 'name' is
                         the corresponding access key in the YAML file.
        """
        super(relay, self).__init__(callevery=callevery)

        # Meta
        self.switches = switches
        self.ymlfile = ymlfile
        self.lastmodified = 0.

    def read(self):
        # Check if there are changes to the read
        filehaschanged = os.stat(self.ymlfile).st_mtime != self.lastmodified
        # Update lastmodified timestamp
        if filehaschanged:
            self.lastmodified = os.stat(self.ymlfile).st_mtime
        else:
            return

        # Read from file
        with open(self.ymlfile, 'r') as f:
            update = yaml.load(f)

        # Update switches
        for switchname, switchvar in self.switches.items():
            # Fetch
            if switchname in update.keys():
                # Check if update needs to be eval-ed
                if isinstance(update[switchname], str) and update[switchname].startswith('np.'):
                    switchvarval = eval(update[switchname])
                else:
                    switchvarval = getattr(np, config.floatX)(update[switchname])
                # Set switch variable
                switchvar.set_value(switchvarval)
        return

    def __call__(self, *args, **kwargs):
        self.read()


class printer(callback):
    """Class to print training/validation outputs."""
    def __init__(self, monitors, textlogger=None, callevery=1):
        """
        :type monitors: list
        :param monitors: Training monitors that return a string.

        :type textlogger: logger
        :param textlogger: Logfile to print to.
        """
        super(printer, self).__init__(callevery=callevery)

        # Make sure monitors is a list
        monitors = list(monitors)
        # Convert all functions in monitors to smart functions
        monitors = [smartfunc(monitor, ignorekwargssilently=True) for monitor in monitors]

        # Attach meta
        self.monitors = monitors
        self.textlogger = textlogger
        # Previous print
        self.prevprint = ''

    def __call__(self, *args, **kwargs):
        # Get monitor outputs
        monitorout = [monitor(**kwargs) for monitor in self.monitors]
        # Get rid of None's if there are any
        monitorout = [mout for mout in monitorout if mout is not None]
        # Build message
        msg = ''.join(monitorout)
        self.prevprint = msg
        # Print message
        print(msg)
        # Log printed message to file
        if self.textlogger is not None:
            self.textlogger(msg)


class steplogger(callback):
    """Log to an internal datastructure, one that can be written out to a pickled file."""
    def __init__(self, fieldnames, filename=None, callevery=1):
        """
        :type fieldnames: tuple or list
        :param fieldnames: Fields to log. The __call__ method expects all entries in fieldnames as keywords.

        :type filename: str
        :param filename: Where to read from / write to.
        """
        super(steplogger, self).__init__(callevery=callevery)

        # Meta
        self.fieldnames = list(set(fieldnames))
        self.logdict = {fieldname: [] for fieldname in self.fieldnames}
        self.filename = filename

    def __call__(self, *args, **kwargs):
        assert not args, "Steplogger requires keyword arguments."

        if not kwargs:
            return

        # Repalce kwargs with a working kwargs. If the user missed an entry, replace with None.
        tolog = {fieldname: None for fieldname in self.fieldnames}
        tolog.update({kw: kwval for kw, kwval in kwargs.items() if kw in self.fieldnames})

        # Make sure there aren't any unexpected arguments to log
        assert set(tolog.keys()) == set(self.fieldnames), "Found key(s) matching none of the set fieldnames"

        # Log all fields
        for fieldname, fieldvalue in tolog.items():
            self.logdict[fieldname].append(fieldvalue)

    def log(self, **kwargs):
        self(**kwargs)

    def __getitem__(self, item):
        assert item in self.fieldnames, "No fields matching {} found.".format(item)
        return self.logdict[item]

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename

        assert filename is not None, "No filename given."
        # Pickle
        pickle(self.logdict, filename)

    def read(self, filename=None):
        if filename is None:
            filename = self.filename
        else:
            self.filename = filename

        assert filename is not None, "No filename given."
        self.logdict = unpickle(filename)


class plotter(callback):
    """Class to broadcast time series to a Bokeh server."""
    def __init__(self, linenames, sessionname=None, colors=None, xaxis='iterations', callevery=1):
        """
        :type linenames: list or tuple
        :param linenames: Names of the time series for the Bokeh server to plot. The call method must have the names
                          in the list as keyword arguments.

        :type sessionname: str
        :param sessionname: Name of the Bokeh session

        :type xaxis: str
        :param xaxis: What goes in the x axis ('iterations' or 'epochs')
        """
        super(plotter, self).__init__(callevery=callevery)

        # Meta
        self.linenames = list(linenames)
        self.sessionname = "PlotterID-{}".format(id(self)) if sessionname is None else sessionname
        self.colors = colors if colors is not None else [None,] * len(self.linenames)
        self.xaxis = xaxis

        # Init plot server
        bplt.output_server(self.sessionname)

        # Build figure
        self.figure = bplt.figure()
        self.figure.xaxis.axis_label = self.xaxis

        # Make lines in figure
        for name, color in zip(self.linenames, self.colors):
            if color is not None:
                self.figure.line(x=[], y=[], name=name, color=color, legend=name)
            else:
                self.figure.line(x=[], y=[], name=name, legend=name)

        bplt.show(self.figure)

        # Make list of renderers and datasources
        self.renderers = {name: self.figure.select(dict(name=name)) for name in self.linenames}
        self.datasources = {name: self.renderers[name][0].data_source for name in self.linenames}

    def __call__(self, *args, **kwargs):
        assert all([linename in kwargs.keys() for linename in self.linenames]), "Line names must be in dict of keyword " \
                                                                                "arguments."

        for linename in self.linenames:
            # Fetch from kwargs
            inp = kwargs[linename]
            # Parse
            if isinstance(inp, (tuple, list)):
                assert len(inp) == 2
                xval, yval = inp
            else:
                yval = inp
                if self.xaxis == 'iterations':
                    xval = kwargs['iternum']
                elif self.xaxis == 'epochs':
                    xval = kwargs['epoch']
                elif self.xaxis in kwargs.keys():
                    # Easter egg
                    xval = kwargs[self.xaxis]
                else:
                    raise NotImplementedError("Unknown token for x-axis: {}. Correct tokens are: "
                                              "'iterations' and 'epochs'".format(self.xaxis))

            self.datasources[linename].data["y"].append(yval)
            self.datasources[linename].data["x"].append(xval)

            # Send to server
            bplt.cursession().store_objects(self.datasources[linename])


class timemachine(callback):
    def __init__(self, every, units):
        """
        Backup parameters every `every` `units` units (tehe).

        :type every: int
        :param every: Backup Frequency

        :type units: str
        :param units: Backup frequency units.
                      Possible keys:
                        - 'iterations', 'iteration', 'iter', 'iters', 'i'
                        - 'epochs', 'epoch', 'eps', 'e'
        """

        super(timemachine, self).__init__(callevery=1)

        # Meta
        self.every = every
        self.units = 'i' if units in ['iterations', 'iteration', 'iter', 'iters', 'i'] else 'e'
        self.baggage = {}

    def __call__(self, *args, **kwargs):
        assert not args
        # Make sure the required arguments are in kwargs
        assert all([requiredkw in kwargs.keys() for requiredkw in ['model', 'iternum', 'epoch']])
        self.save(**kwargs)

    def save(self, **kwargs):
        # Do the dirty work
        iternum = kwargs['iternum']
        model = kwargs['model']
        epoch = kwargs['epoch']

        if self.units == 'i':
            if 'lastiter' not in self.baggage.keys():
                # This gets executed only for the first time this function is called.
                self.baggage['lastiter'] = iternum
                model.save(nameflgs='--iter-{}-routine'.format(iternum))
            else:
                if iternum > self.baggage['lastiter']:
                    self.baggage['lastiter'] = iternum
                    # New iteration - save parameters
                    model.save(nameflgs='--iter-{}-routine'.format(iternum))
        else:
            if 'lastepoch' not in self.baggage.keys():
                # This gets executed only for the first time this function is called.
                self.baggage['lastepoch'] = epoch
                model.save(nameflags='--epoch-{}-routine'.format(epoch))
            else:
                if epoch > self.baggage['lastepoch']:
                    self.baggage['lastepoch'] = epoch
                    # New epoch - save parameters
                    model.save(nameflags='--epoch-{}-routine'.format(epoch))

class caller(callback):
    def __init__(self, function, callevery=1):
        """
        Callback that calls any given function. The function must take keyword arguments as **kwargs.

        :type function: callable
        :param function: Function to call.

        :type callevery: int
        :param callevery: Callback is to be called every `callbackevery` iteration.
        """
        super(caller, self).__init__(callevery=callevery)
        # Meta
        self.function = function

    def __call__(self, *args, **kwargs):
        # Hello from the other si... *kadoosh*
        self.function(**kwargs)


# Yaml to dict reader
def yaml2dict(path):
    with open(path, 'r') as f: readict = yaml.load(f)
    return readict


# Printer factory
def makeprinter(verbosity, extramonitors=None):
    if verbosity >= 4:
        monitors = [batchmonitor, costmonitor, lossmonitor, trEmonitor, gradnormmonitor, updatenormmonitor]
    elif verbosity >= 3:
        monitors = [batchmonitor, costmonitor, lossmonitor, trEmonitor]
    elif verbosity >= 2:
        monitors = [batchmonitor, costmonitor]
    else:
        monitors = []

    if extramonitors is not None:
        extramonitors = pyk.obj2list(extramonitors)
        # Append extra monitors
        monitors.extend(extramonitors)

    # Build printer
    prntr = printer(monitors)
    return prntr

def defaultcallback(verbosity):
    if verbosity > 0:
        return callbacks([makeprinter(verbosity=verbosity)])
    else:
        return callbacks([])

def batchmonitor(batchnum=None):
    if batchnum is not None:
        return "| Batch: {} |".format(batchnum)
    else:
        return ""


def lossmonitor(L=None):
    if L is not None:
        return "| Loss: {0:.7f} |".format(float(L))
    else:
        return ""


def costmonitor(C=None):
    if C is not None:
        return "| Cost: {0:.7f} |".format(float(C))
    else:
        return ""


def trEmonitor(E=None):
    if E is not None:
        return "| Training Error: {0:.3f} |".format(float(E))
    else:
        return ""


def vaEmonitor(vaE=None):
    if vaE is not None:
        return "| Validation Error: {0:.3f} |".format(float(vaE))
    else:
        return ""


def gradnormmonitor(dC=None):
    if dC is not None:
        gradnorm = sum([np.sum(dparam ** 2) for dparam in dC])
        return "| Gradient Norm: {0:.7f} |".format(float(gradnorm))
    else:
        return ""


def exploremonitor(params=None):
    if hasattr(exploremonitor, "prevparams"):
        # Convert params to numerical value
        params = sym2num(params)
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
        exploremonitor.prevparams = sym2num(params)
        # Log lower and upper bounds
        exploremonitor.lb = exploremonitor.prevparams
        exploremonitor.ub = exploremonitor.prevparams
        # Return
        return "| Explored Dimensions: N/A |"


def updatenormmonitor(params):
    if hasattr(updatenormmonitor, "prevparams"):
        # Convert to num
        params = sym2num(params)
        # Compute update norm
        updnorm = sum([np.sum((currparam - prevparam) ** 2)
                       for currparam, prevparam in zip(params, updatenormmonitor.prevparams)])
        # Log
        updatenormmonitor.prevparams = params
        return "| Update Norm: {0:.7f} |".format(float(updnorm))
    else:
        updatenormmonitor.prevparams = sym2num(params)
        return "| Update Norm: N/A |"
    pass


def plotmonitor(E=None):
    # Make a call counter (to count the number of times the function was called)
    if not hasattr(plotmonitor, "callcount"):
        plotmonitor.callcount = 0
    else:
        plotmonitor.callcount += 1

    # Make a variable to store a history of input arguments
    if not hasattr(plotmonitor, "history"):
        plotmonitor.history = np.array([])
        plotmonitor.history = np.append(plotmonitor.history, E)
    else:
        plotmonitor.history = np.append(plotmonitor.history, E)

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