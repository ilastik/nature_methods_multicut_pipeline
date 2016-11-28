__author__ = "nasim.rahaman@iwr.uni-heidelberg.de"
__doc__ = """General script for running networks."""


def pathsy(path):
    """Parse paths."""
    # This file is .../snemi/Scripts/train.py
    thisdirectory = os.path.dirname(__file__)
    # This is the SNEMI directory. path must be relative to this path
    snemihome = os.path.normpath(thisdirectory + '/../')
    # Target path
    outpath = os.path.join(snemihome, path)
    return outpath


def fetchfromdict(dictionary, key, default=None):
    """Try to fetch `dictionary[key]` if possible, return `default` otherwise."""
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


def run(net, trX, **runconfig):
    """Given an Antipasti model object `net` and an Antipasti datafeeder `trX`, configure callbacks and fit model."""
    # Aliases
    ffd = fetchfromdict

    assert 'savedir' in runconfig.keys(), "Backup directory must be provided!"
    print("[+] Saving parameters to {}.".format(runconfig['savedir']))
    net.savedir = pathsy(runconfig['savedir'])

    # Configure logfile
    if ffd(runconfig, 'logfile') is not None:
        print("[+] Logging to file at {}.".format(pathsy(runconfig['logfile'])))
        log = tk.logger(pathsy(runconfig['logfile']))
    else:
        print("[-] Not logging progress.")
        log = None

    # Configure relay
    if ffd(runconfig, 'relayfile') is not None:
        print("[+] Listening to relay file at {}.".format(pathsy(runconfig['relayfile'])))
        relay = tk.relay({'learningrate': net.baggage['learningrate']}, pathsy(runconfig['relayfile']))
    else:
        print("[-] Not listening to relays.")
        relay = None

    print("[+] Building callbacks...")
    # Build callbacks
    if ffd(runconfig, 'live-plot') is not None:
        print("[+] Live plots will be displayed on a Bokeh server on localhost:5006. If this results in an error, "
              "make sure to open a terminal and key in 'bokeh-server' before running this script again.")
        cbs = tk.callbacks([tk.makeprinter(verbosity=5), tk.plotter(linenames=['C', 'L'], colors=['navy', 'firebrick'])])
    else:
        cbs = tk.callbacks([tk.makeprinter(verbosity=5)])

    # Bind textlogger to printer
    cbs.callbacklist[0].textlogger = log

    # Tell fit method whether a weight map will be provided
    if 'wmap' in net.baggage.keys():
        print("[+] Using weight maps.")
        extrargs = {net.baggage['wmap']: -1}
    else:
        print("[-] Not using weight maps.")
        extrargs = {}

    print("[+] Ready to train.")
    # Fit
    res = net.fit(trX=trX, numepochs=800, verbosity=5, backupparams=200, log=log, trainingcallbacks=cbs,
                  extrarguments=extrargs, relay=relay)

    if ffd(runconfig, 'picklejar') is not None:
        print("[+] Pickling run results to {}.".format(pathsy(runconfig['picklejar'])))
        nu.pickle(res, os.path.join(pathsy(runconfig['picklejar']), 'fitlog.save'))
    else:
        print("[-] Not pickling results.")

    print("[+] Done.")

    return net


def plot(net, trX, **plotconfig):
    """Plot intermediate results given a model `net` and a datafeeder `trX`."""

    # Glob params for smoother UI
    plotconfig['params'] = glob.glob(pathsy(plotconfig['params']))[0]
    # plotconfig['params'] could be a directory. If that's the case, select the most recent parameter file and load
    if os.path.isdir(plotconfig['params']):
        print("[-] Given parameter file is a directory. Will fetch the most recent set of parameters.")
        # It's a dir
        # Get file name of the most recent file
        ltp = sorted(os.listdir(plotconfig['params']))[-1]
        parampath = os.path.join(plotconfig['params'], ltp)
    else:
        # It's a file
        parampath = plotconfig['params']
        pass

    print("[+] Loading parameters from {}.".format(parampath))

    # Load params
    net.load(parampath)

    # Get batches from feeders
    batches = [trX.next() for _ in range(plotconfig['numbatches'])]

    for n, batch in enumerate(batches):
        print("[+] Evaluating batch {}...".format(n))
        bX, bY = batch[0:2]

        ny = net.y.eval({net.x: bX})
        vz.printensor2file(bX, savedir=plotconfig['plotdir'], mode='image', nameprefix='RD{}--'.format(n))
        vz.printensor2file(bY, savedir=plotconfig['plotdir'], mode='image', nameprefix='GT{}--'.format(n))
        vz.printensor2file(ny, savedir=plotconfig['plotdir'], mode='image', nameprefix='PR{}--'.format(n))

    print("[+] Plotted images to {}.".format(plotconfig['plotdir']))

    print("[+] Done.")


if __name__ == '__main__':
    print("[+] Initializing...")
    import argparse
    import yaml
    import sys
    import os
    import imp
    import glob

    # Parse arguments
    parsey = argparse.ArgumentParser()
    parsey.add_argument("configset", help="Configuration file.")
    parsey.add_argument("--device", help="Device to use (overrides configuration file).", default=None)
    args = parsey.parse_args()

    # Load configuration dict
    with open(args.configset) as configfile:
        config = yaml.load(configfile)

    print("[+] Using configuration file from {}.".format(args.configset))

    # Read which device to use
    if args.device is None:
        device = config['device']
    else:
        device = args.device

    assert device is not None, "Please provide the device to be used as a bash argument " \
                               "(e.g.: python train.py /path/to/config/file.yml --device gpu0)"

    print("[+] Using device {}.".format(device))

    # Import shit
    from theano.sandbox.cuda import use
    use(device)

    # Add Antipasti to path
    sys.path.append(pathsy('Antipasti'))
    import Antipasti.trainkit as tk
    import Antipasti.netutils as nu
    import Antipasti.vizkit as vz

    print("[+] Importing model and datafeeders...")
    # Import model
    model = imp.load_source('model', pathsy(config['modelpath']))
    # Import datafeeder
    dpl = imp.load_source('dataplate', pathsy(config['dplpath']))

    print("[+] Building network...")
    # Build network
    net = model.build(**config['buildconfig'])

    print("[+] Fetching feeders with configuration at {}.".format(config['dataconf']))
    # Fetch datafeeders
    trX = dpl.fetchfeeder(config['dataconf'])

    if 'runconfig' in config.keys():
        print("[+] Ready to run.")
        # Run training
        run(net, trX, **config['runconfig'])
    elif 'plotconfig' in config.keys():
        print("[+] Ready to plot.")
        # Plot
        plot(net, trX, **config['plotconfig'])
