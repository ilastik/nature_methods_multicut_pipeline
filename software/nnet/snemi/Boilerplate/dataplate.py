__doc__ = """Boilerplate for SNEMI."""

import os

def pathsy(path):
    """Parse paths."""
    # This file is .../snemi/Scripts/train.py
    thisdirectory = os.path.dirname(__file__)
    # This is the SNEMI directory. path must be relative to this path
    snemihome = os.path.normpath(thisdirectory + '/../')
    # Target path
    outpath = os.path.join(snemihome, path)
    return outpath

# Add to path
import sys
sys.path.append(pathsy('Antipasti'))

import Antipasti.trainkit as tk
import Antipasti.prepkit as pk
import Antipasti.netdatautils as ndu
import Antipasti.netdatakit as ndk

# Project specific imports
sys.path.append(pathsy('Boilerplate'))
import prepfunctions


def ffd(dictionary, key, default=None):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


def path2dict(path):
    if isinstance(path, str):
        return tk.yaml2dict(pathsy(path))
    elif isinstance(path, dict):
        return path
    else:
        raise NotImplementedError


def buildpreptrains(**prepconfig):
    """Build train of preprocessing functions."""

    # Fetch preprocessing fns
    pf = prepfunctions.prepfunctions()

    # There are 3 preptrains to be built: X, Y and XY.
    ptX = pk.preptrain([pk.normalizebatch(), pf['time2channel' + ('2D' if prepconfig['is2D'] else '')]])
    
    # Select mode
    if ffd(prepconfig, 'make-wmap', False):
        ptY = pk.preptrain([pf['time2channel' + ('2D' if prepconfig['is2D'] else '')], pf['seg2classlabels'](),
                            pk.cast('float32')])
    else:
        ptY = pk.preptrain([pf['time2channel'], pf['seg2membrane'](), pf['disttransform'](**prepconfig['edt-params']),
                            pk.cast('float32')])

    # Joint prepfunctions
    ptXY = pk.preptrain([])

    if prepconfig['elastic-transform']:
        ptXY.append(pf['elastictransform'](**prepconfig['elastic-transform-params']))

    if prepconfig['random-flip-z']:
        ptXY.append(pf['randomflipz']())

    if prepconfig['random-rotate']:
        ptXY.append(pf['randomrotate']())

    if prepconfig['random-flip']:
        ptXY.append(pf['randomflip']())

    if ffd(prepconfig, 'make-wmap', False):
        ptXY.append(pf['wmapmaker'](**prepconfig['wmap-maker-params']))

    return {'X': ptX, 'Y': ptY, 'XY': ptXY}


def load(**loadconfig):
    """Load data to RAM."""

    # Load raw data
    volX = ndu.fromh5(path=pathsy(loadconfig['X']['path']), datapath=loadconfig['X']['h5path'],
                      dataslice=eval(loadconfig['X']['dataslice']))

    # Transpose volX to CREMI-esque format
    volX = volX.transpose(2, 1, 0)
    # -----------------------------------------------------------------------------------------------

    # Load ground truth
    volY = ndu.fromh5(path=pathsy(loadconfig['Y']['path']), datapath=loadconfig['Y']['h5path'],
                      dataslice=eval(loadconfig['Y']['dataslice']))

    # Transpose volX to CREMI-esque format
    volY = volY.transpose(2, 1, 0)
    # -----------------------------------------------------------------------------------------------

    assert volX.shape == volY.shape, "Shape mismatch: volX.shape = {} " \
                                     "but volY.shape = {}.".format(volX.shape, volY.shape)

    # Make dict and return
    return {'X': volX, 'Y': volY}


def fetchfeeder(dataconf):
    """Fetches Antipasti feeder given data configuration dict."""

    # Convert to dictionary
    dataconf = path2dict(dataconf)

    # Load datasets
    datasets = load(**dataconf['loadconfig'])

    # Check from config if data is 2D
    is2D = dataconf['nhoodsize'][0] == 1

    # Build preptrains
    preptrains = buildpreptrains(is2D=is2D, **dataconf['prepconfig'])

    # Build feeders
    gt = ndk.cargo(data=datasets['Y'],
                   axistags='kij', nhoodsize=dataconf['nhoodsize'], stride=dataconf['stride'], ds=dataconf['ds'],
                   batchsize=dataconf['batchsize'], window=['x', 'x', 'x'], preptrain=preptrains['Y'])

    rd = gt.clonecrate(data=datasets['X'], syncgenerators=True)
    rd.preptrain = preptrains['X']

    # Zip feeders and append preptrain
    rdgt = ndk.feederzip([rd, gt], preptrain=preptrains['XY'])

    # Return
    return rdgt


def test(dataconf):
    import Antipasti.vizkit as vz

    # Convert to dict and make sure it checks out
    dataconf = path2dict(dataconf)
    assert 'plotdir' in dataconf.keys(), "The field 'plotdir' must be provided for printing images."

    print("[+] Building datafeeders...")
    # Make feeder
    feeder = fetchfeeder(dataconf)

    print("[+] Fetching datafeeders from file...")
    # Fetch from feeder
    batches = feeder.next()

    # Print
    for n, batch in enumerate(batches):
        print("Printing object {} of shape {} to file...".format(n, batch.shape))
        vz.printensor2file(batch, savedir=pathsy(dataconf['plotdir']), mode='image', nameprefix='N{}--'.format(n))

    print("[+] Done!")

if __name__ == '__main__':

    import argparse
    parsey = argparse.ArgumentParser()
    parsey.add_argument("dataconf", help="Data Configuration file.")
    args = parsey.parse_args()

    # Test
    test(args.dataconf)
