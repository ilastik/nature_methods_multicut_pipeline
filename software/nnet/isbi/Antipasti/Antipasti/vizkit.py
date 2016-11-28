from itertools import product

from matplotlib import pyplot as plt
from numpy import round_
from numpy.core.umath import sqrt, ceil
from scipy.misc import imsave

import netutils as nu

__author__ = 'nasimrahaman'


def plot2dconvfilters(W):
    # Init
    plt.ioff()
    f = plt.figure()
    k = 1
    # Plot
    for i, j in product(range(W.shape[0]), range(W.shape[1])):
        f.add_subplot(W.shape[0], W.shape[1], k)
        k += 1
        plt.imshow(W[i, j, :, :])
        plt.axis('off')


def plotwoimages(im1, im2):
    # Init
    plt.ioff()
    f = plt.figure()
    # Plot im1
    f.add_subplot(2, 1, 1)
    plt.imshow(im1)
    # Plot im2
    f.add_subplot(2, 1, 2)
    plt.imshow(im2)


def plotfeaturemaps(y):
    # Init
    plt.ioff()
    f = plt.figure()
    # Determine Number of Features
    numfeatures = y.shape[1]
    # Determine Optimal Split
    sx = round(sqrt(numfeatures))
    sy = ceil(numfeatures / sx)
    # Plot
    for k in range(numfeatures):
        f.add_subplot(sy, sx, k)
        plt.imshow(y[0, k, :, :])
        plt.axis('off')


def printensor2file(tensor, plotfun=plt.imshow, savedir=None, channels=None, batches=None, interpolation='none',
                    mode='plot', nameprefix='', **savefigkwargs):
    # Remember location for next call
    if hasattr(printensor2file, "saveloc"):
        savedir = printensor2file.savedir
    else:
        assert savedir is not None, "Save location must be given if the function is being called for the first time " \
                                    "after being defined."
        printensor2file.savedir = savedir

    if channels is None:
        channels = range(tensor.shape[(1 if tensor.ndim == 4 else 2)])

    if batches is None:
        batches = range(tensor.shape[0])

    if tensor.ndim == 5:
        for batch in batches:
            for channel in channels:
                for T in range(tensor.shape[1]):
                    if mode == 'image':
                        imsave(name=savedir + nameprefix +
                                    "batch-{}--channel-{}--T-{}".format(batch, channel, T) + '.png',
                               arr=tensor[batch, T, channel, ...])
                    elif mode == 'plot':
                        f = plt.figure()
                        nu.smartfunc(plotfun, ignorekwargssilently=True)(tensor[batch, T, channel, ...],
                                                                         interpolation=interpolation)
                        f.savefig(savedir + nameprefix + "batch-{}--channel-{}--T-{}".format(batch, channel, T),
                                  **savefigkwargs)
                        plt.close(f)
    elif tensor.ndim == 4:
        for batch in batches:
            for channel in channels:
                if mode == 'image':
                    imsave(name=savedir + nameprefix + "batch-{}--channel-{}".format(batch, channel) + '.png',
                           arr=tensor[batch, channel, ...])
                elif mode == 'plot':
                    f = plt.figure()
                    nu.smartfunc(plotfun, ignorekwargssilently=True)(tensor[batch, channel, ...],
                                                                     interpolation=interpolation)
                    f.savefig(savedir + nameprefix + "batch-{}--channel-{}".format(batch, channel), **savefigkwargs)
                    plt.close(f)
