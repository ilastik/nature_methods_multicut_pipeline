import glob

import theano as th
import theano.tensor as T
import numpy as np

import Antipasti.netkit as nk
import Antipasti.netarchs as na
import Antipasti.archkit as ak
import Antipasti.netools as ntl
import Antipasti.netrain as nt


def pathsy(path):
    """Parse paths."""
    # This file is .../snemi/Scripts/train.py
    thisdirectory = os.path.dirname(__file__)
    # This is the SNEMI directory. path must be relative to this path
    snemihome = os.path.normpath(thisdirectory + '/../')
    # Target path
    outpath = os.path.join(snemihome, path)
    return outpath


# Define shortcuts
# Convlayer with ELU
cl = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                     activation=ntl.elu())

# Convlayer without activation
cll = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize)

# Convlayer with Sigmoid
cls = lambda fmapsin, fmapsout, kersize: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout, kersize=kersize,
                                                      activation=ntl.sigmoid())

# Strided convlayer with ELU (with autopad)
scl = lambda fmapsin, fmapsout, kersize, padding=None: nk.convlayer(fmapsin=fmapsin, fmapsout=fmapsout,
                                                                    kersize=kersize,
                                                                    stride=[2, 2], activation=ntl.elu(),
                                                                    padding=padding)

# Strided 3x3 pool layerlayertrain or Antipasti.netarchs.layertrainyard
spl = lambda: nk.poollayer(ds=[3, 3], stride=[2, 2], padding=[1, 1])

# Strided 3x3 mean pool layer
smpl = lambda ds=(2, 2): nk.poollayer(ds=list(ds), poolmode='mean')

# 2x2 Upscale layer
usl = lambda us=(2, 2): nk.upsamplelayer(us=list(us))

# 2x2 Upscale layer with interpolation
iusl = lambda us=(2, 2): nk.upsamplelayer(us=list(us), interpolate=True)

# Batch-norm layer
bn = lambda numinp=None: (nk.batchnormlayer(2, 0.9) if numinp is None else
                          nk.batchnormlayer(2, 0.9, inpshape=[None, numinp, None, None]))

# Softmax
sml = lambda: nk.softmax(dim=2)

# Identity
idl = lambda: ak.idlayer()

# Replicate
repl = lambda numrep: ak.replicatelayer(numrep)

# Merge
merl = lambda numbranch: ak.mergelayer(numbranch)

# Split in half
sptl = lambda splitloc: ak.splitlayer(splits=splitloc, dim=2, issequence=False)

# Dropout layer
drl = lambda p=0.5: nk.noiselayer(noisetype='binomial', p=p)

# Addition layer
adl = lambda numinp: ak.addlayer(numinp, dim=2, issequence=False)

# Circuit layer
crcl = lambda circuit: ak.circuitlayer(circuit, dim=2, issequence=False)

# Parallel tracks
trks = lambda *layers: na.layertrainyard([list(layers)])

lty = lambda ty: na.layertrainyard(ty)


def inceptionize(streams):
    # Compute number of streams
    numstreams = len(streams)
    # Multiply
    module = na.layertrainyard([streams])
    # Build replicate and merge layers
    rep = ak.replicatelayer(numstreams)
    mer = ak.mergelayer(numstreams)
    # Build and return inception module
    return rep + module + mer


def binarizingbce(ist, soll, wmap=None, coeff=1.):
    # Get the usual binary cross entropy loss
    Lbce = nt.bce(ist, soll, wmap=wmap)
    # Get binarizing term
    binterm = coeff * (ist * (1. - ist)).mean()
    # Add to get the final loss
    L = Lbce + binterm
    return L


def build(numinp=3, numout=3, wmap=False, binarize=False, parampath=None):
    """Define Antipasti model and build its theano graph."""

    # Build the networkdparam), 0., dparam) for dparam in dC]
    # Return
    # --- a1 --- b1 --- --- c1 --- d1 --- d2 --- c2 --- --- b1 --- a1 ---
    #                  |                               |
    #                   ------------- id --------------

    print("[+] Building ICv1 with {} inputs and {} outputs.".format(numinp, numout))

    a1 = cl(numinp, 32, [9, 9]) + drl() + cl(32, 48, [9, 9])

    b1 = scl(48, 128, [7, 7]) + drl() + \
         inceptionize([cl(128, 64, [3, 3]) + cl(64, 64, [1, 1]), cl(128, 64, [5, 5]) + cl(64, 64, [3, 3])]) + \
         cl(128, 160, [3, 3])

    c1 = inceptionize([cl(160, 64, [5, 5]) + spl(), scl(160, 64, [3, 3]) + cl(64, 96, [1, 1])]) + \
         cl(160, 160, [3, 3]) + drl() + \
         inceptionize([cl(160, 100, [7, 7]), cl(160, 48, [5, 5]) + cl(48, 48, [1, 1]),
                       cl(160, 64, [3, 3]) + cl(64, 64, [1, 1])]) + \
         cl(212, 240, [3, 3])

    d1 = inceptionize([cl(240, 192, [1, 1]) + spl(), scl(240, 512, [3, 3])]) + cl(704, 1024, [3, 3])

    d2 = drl() + inceptionize([cl(1024, 384, [3, 3]) + cl(384, 200, [3, 3]), cl(1024, 260, [1, 1]),
                               cl(1024, 384, [5, 5]) + cl(384, 200, [1, 1])]) + \
         cl(660, 512, [3, 3]) + \
         inceptionize([cl(512, 60, [7, 7]), cl(512, 180, [3, 3])]) + \
         usl()

    c2 = drl() + cl(240, 200, [3, 3]) + \
         inceptionize([cl(200, 140, [3, 3]) + cl(140, 80, [3, 3]), cl(200, 140, [5, 5]) + cl(140, 80, [5, 5])]) + \
         cl(160, 160, [5, 5]) + \
         usl()

    b2 = drl() + cl(320, 128, [5, 5]) + \
         inceptionize([cl(128, 60, [9, 9]) + cl(60, 48, [5, 5]), cl(128, 72, [5, 5]) + cl(72, 48, [5, 5])]) + \
         cl(96, 60, [5, 5]) + \
         cl(60, 48, [3, 3]) + \
         usl()

    a2 = drl() + cl(48, 32, [9, 9]) + cl(32, 16, [5, 5]) + cl(16, 16, [3, 3]) + cls(16, numout, [1, 1])

    # Putting it together
    interceptorv1 = a1 + b1 + repl(2) + (c1 + d1 + d2 + c2) * idl() + merl(2) + b2 + a2
    interceptorv1.feedforward()

    # Load parameters
    if parampath is not None:
        parampath = glob.glob(pathsy(parampath))[0]
        print("[+] Loading pretrained parameters from {}.".format(parampath))
        interceptorv1.load(parampath)
    else:
        print("[-] Not loading pretrained parameters.")

    # Set up cost function and optimizer
    interceptorv1.baggage["learningrate"] = th.shared(value=np.float32(0.0002))

    if binarize:
        print("[+] Using binarizing loss.")
        costmethod = binarizingbce
    else:
        print("[+] Not using binarizing loss.")
        costmethod = 'bce'

    if wmap:
        print("[+] Using weight maps.")
        # Include weight map
        interceptorv1.baggage["wmap"] = T.tensor4()
        interceptorv1.cost(method=costmethod, wmap=interceptorv1.baggage['wmap'], regterms=[(2, 0.0005)])
    else:
        print("[-] Not using weight maps.")
        # Don't include weight map
        interceptorv1.cost(method=costmethod, regterms=[(2, 0.0005)])

    interceptorv1.getupdates(method='adam', learningrate=interceptorv1.baggage['learningrate'])

    return interceptorv1