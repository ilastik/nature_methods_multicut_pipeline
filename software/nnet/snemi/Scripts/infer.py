__doc__ = """Inference Engine for CREMI and with some luck, for SNEMI."""
__author__ = "nasim.rahaman at iwr.uni-heidelberg.de"

import multiprocessing as mp
import Queue as q
import sys
import os
import yaml
import imp
import argparse
import time
import datetime

from random import choice, shuffle
from argparse import Namespace
from itertools import product


import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def pathsy(path):
    """Parse paths."""
    # This file is .../snemi/Scripts/train.py
    thisdirectory = os.path.dirname(__file__)
    # This is the SNEMI directory. path must be relative to this path
    snemihome = os.path.normpath(thisdirectory + '/../')
    # Target path
    outpath = os.path.join(snemihome, path)
    return outpath


# Imports without theano dependency
sys.path.append(pathsy('Antipasti'))
import Antipasti.prepkit as pk


class worker(mp.Process):
    def __init__(self, workerconfig, jobq, resultq, lock=None):
        """
        Worker class.

        :param workerconfig: Configuration for the worker

        :param jobq: Job queue. Must contain dictionaries precisely specifying the job.
                     Example dictionary:
                        {'dataset': 'A', 'slice': (slice(0, None, 2), ...)}


        :param resultq: Result queue. The results are written to this queue and sent to the supervisor.
                        Example dictionary:
                            {'dataset': 'A', 'slice': (slice(0, None, 2), ...), 'payload': np.ndarray(...)}

        :param lock: Lock to use while reading from the job queue (to prevent racing conditions).
        """
        # Init superclass
        super(worker, self).__init__()

        # Meta
        self.workerconfig = workerconfig
        self.jobq = jobq
        self.resultq = resultq
        self.lock = lock

        self.verbose = True
        self.logger = None
        self.configurelogger()

        # Load volumes
        # Try to get from workerconfig if possible (to not blow up memory usage)
        self.datasets = self.workerconfig['datasets'] if 'datasets' in self.workerconfig.keys() else None
        # Tough luck, load shit to RAM
        if self.datasets is None:
            self.load()

        # Define placeholders for augmentation and deaugmentation functions
        self.augfunc = None
        self.deaugfunc = None

        # Storage
        self.baggage = {}

    def load(self):
        """Loads volumes to RAM."""
        raise NotImplementedError

    def genrandaug(self, batchshape):
        """
        Generates random augmentation and deaugmentation callables. This function populates the `augfunc` and `deaugfunc`
        fields of this class.

        `augfunc` and `deaugfunc` take in a 4D tensor of shape `(batchsize, numslices, numrows, numcols)`. Now
        `batchsize` is important for GPU efficiency, but for the data-augmentation, this requires a function array of
        `batchsize` augmentation and deaugmentation functions, one for each element in the batch.
        """

        def pairfactory():
            """Factory function that makes a pair of augmentation and deaugmentation functions."""
            # Have the random augmentations hashed by a dictionary. This dict will be used by both aug and deaug.
            # Get batchshape after padding

            if batchshape is not None:
                paddedbatchshape = batchshape[0:2] + \
                                   tuple([bs + 2 * self.workerconfig['daconfig']['pad'] for bs in batchshape[2:]])
            else:
                paddedbatchshape = None

            hshdict = {'fliplr': choice([True, False]),
                       'flipud': choice([True, False]),
                       'rot90': choice([0, 1, 2, 3]),
                       'transpose': choice([True, False]),
                       'flipz': choice([True, False]),
                       'et(dx, dy)': ((np.random.uniform(-1, 1, paddedbatchshape[2:]),
                                       np.random.uniform(-1, 1, paddedbatchshape[2:])) if batchshape is not None else None),
                       'et(rng)': np.random.randint(-100, 100)}

            def _et(img, dxdy=None, rngseed=None, invert=False, sigma=50., alpha=2000.):
                # img is actually an array of im's.
                imshape = img.shape[1:]

                if batchshape is not None:
                    assert imshape == paddedbatchshape[2:], "Shape inconsistency. Incoming image shape is {}, but the " \
                                                            "provided batchshape[2:] after " \
                                                            "padding is {}.".format(imshape, paddedbatchshape[2:])

                if dxdy is not None:
                    # First, try to get dxdy
                    dx, dy = dxdy
                    dx, dy = (alpha * dx, alpha * dy)
                else:
                    # If that was not possible, make a rng
                    rng = np.random.RandomState(seed=rngseed)
                    dx = rng.uniform(-1, 1, imshape) * alpha
                    dy = rng.uniform(-1, 1, imshape) * alpha

                # Smooth dx and dy
                sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
                sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
                # Inversion sign
                invsgn = -1. if invert else 1.
                # Make meshgrid
                x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
                # Distort meshgrid indices (invert if required)
                distinds = (y + invsgn * sdy).reshape(-1, 1), (x + invsgn * sdx).reshape(-1, 1)
                # Map cooordinates from image to distorted index set
                timg = np.array([map_coordinates(im, distinds, mode='reflect', order=1).reshape(imshape) for im in img])

                return timg

            def augfactory():
                """Function to convert img to a batch of augmented images for the network to process."""
                # Make a list of lambda functions that apply a certain transformation. This list will be shuffled
                # before augmenting the image, causing the augmentations to be applied in a random order.

                augs = []
                # Pad
                pad = self.workerconfig['daconfig']['pad']
                augs.append(lambda imag: (np.array([np.pad(im, ((pad, pad), (pad, pad)), 'reflect') for im in imag])
                                          if self.workerconfig['daconfig']['pad'] else imag))
                # fliplr
                augs.append(lambda imag: (np.array([np.fliplr(im) for im in imag]) if hshdict['fliplr'] else imag))
                # flipud
                augs.append(lambda imag: (np.array([np.flipud(im) for im in imag]) if hshdict['flipud'] else imag))
                # rot90
                augs.append(lambda imag: (np.array([np.rot90(im, hshdict['rot90']) for im in imag])
                                          if hshdict['flipud'] else imag))
                # flipz
                augs.append(lambda imag: (imag[::-1, ...] if hshdict['flipz'] else imag))
                # transpose
                augs.append(lambda imag: (imag.transpose(0, 2, 1) if hshdict['transpose'] else imag))
                # Elastic transform
                augs.append(lambda imag: _et(imag, dxdy=hshdict['et(dx, dy)'],
                                             rngseed=hshdict['et(rng)'], invert=False,
                                             sigma=self.workerconfig['daconfig']['et']['sigma'],
                                             alpha=self.workerconfig['daconfig']['et']['alpha']))

                # Make preptrain
                return pk.preptrain(augs)

            def deaugfactory():
                """Function to undo the augmentation and convert a batch to an image."""
                # Inverse augmentations
                deaugs = []

                # Crop
                pad = self.workerconfig['daconfig']['pad']
                deaugs.append(lambda imag: (imag[:, pad:-pad, pad:-pad] if pad else imag))
                # fliplr
                deaugs.append(lambda imag: (np.array([np.fliplr(im) for im in imag]) if hshdict['fliplr'] else imag))
                # flipud
                deaugs.append(lambda imag: (np.array([np.flipud(im) for im in imag]) if hshdict['flipud'] else imag))
                # rot90
                deaugs.append(lambda imag: (np.array([np.rot90(im, 4-hshdict['rot90']) for im in imag])
                                          if hshdict['flipud'] else imag))
                # flipz
                deaugs.append(lambda imag: (imag[::-1, ...] if hshdict['flipz'] else imag))
                # transpose
                deaugs.append(lambda imag: (imag.transpose(0, 2, 1) if hshdict['transpose'] else imag))
                # Elastic transform
                deaugs.append(lambda imag: _et(imag, dxdy=hshdict['et(dx, dy)'],
                                               rngseed=hshdict['et(rng)'], invert=True,
                                               sigma=self.workerconfig['daconfig']['et']['sigma'],
                                               alpha=self.workerconfig['daconfig']['et']['alpha']))

                # Deaugmentations are to applied in reversed order.
                deaugs.reverse()

                # Build preptrain and return
                return pk.preptrain(deaugs)

            # Make augmentations and deaugmentations
            aug = augfactory()
            deaug = deaugfactory()

            return aug, deaug

        def funarrayfactory():
            """Makes an array of random augmentation functions, one for each element in batch."""
            # Get batchsize
            bs = batchshape[0]
            # Make as many function pairs
            augfns, deaugfns = zip(*[pairfactory() for _ in range(bs)])

            def augment(batch):
                # Assertions
                # assert batch.shape == batchshape, "Shape inconsistency."
                # Augment
                outbatch = np.array([augfn(img) for img, augfn in zip(batch, augfns)])
                # Return
                return outbatch

            def deaugment(batch):
                # Assertions
                # assert batch.shape == batchshape, "Shape inconsistency."
                # Deaugment
                outbatch = np.array([deaugfn(img) for img, deaugfn in zip(batch, deaugfns)])
                # Return
                return outbatch

            return augment, deaugment

        # Write to object fields
        self.augfunc, self.deaugfunc = funarrayfactory()

    def fetchtensor(self, jobs):
        """
        Given a list of jobs (a job is a dictionary, see docstring for parameter `jobq` in __init__), fetch the
        corresponding tensor.
        """
        tensor = np.array([self.datasets[job['dataset']][job['slice']] for job in jobs])
        return tensor

    def print_(self, msg):
        if self.verbose:
            print("Process {}: {}".format(os.getpid(), msg))

        if self.logger is not None:
            self.logger("Process {}: {}".format(os.getpid(), msg))

    def configurelogger(self):
        logfile = self.workerconfig['logfile'] if 'logfile' in self.workerconfig.keys() else None
        if logfile is not None:
            self.logger = logger(logfile)

    def build(self):
        """Build model"""
        modelfile = imp.load_source('sierpinskinet', pathsy(self.workerconfig['modelpath']))

        self.print_("[+] Building Model...")
        network = modelfile.build(**self.workerconfig['buildparams'])

        self.print_("[+] Compiling Inference Function...")
        network.compile(what='inference')
        return network

    def waitforpid(self):
        if 'waitforpid' in self.workerconfig.keys():
            while True:
                if os.path.exists('/proc/{}'.format(self.workerconfig['waitforpid'])):
                    self.print_("Waiting for PID {}.".format(self.workerconfig['waitforpid']))
                    time.sleep(10)
                else:
                    self.print_("Done waiting for PID {}.".format(self.workerconfig['waitforpid']))
                    break

    def run(self):
        # Wait for device
        self.waitforpid()
        # Import theano and bind it to the GPU
        if 'gpu' in self.workerconfig['device']:
            self.print_("[+] Trying to initialize GPU device {}.".format(self.workerconfig['device']))
            from theano.sandbox.cuda import use
            use(self.workerconfig['device'])
        else:
            self.print_("[-] Not using GPU. The device is set to {}.".format(self.workerconfig['device']))

        self.print_("[+] Importing theano...")
        import theano as th

        try:
            # Build network
            network = self.build()
        except Exception as e:
            print("[-] Exception raised while building network. The error message is as follows: {}".format(e.message))
            # Send poison pill and call it a day
            self.resultq.put(None)
            self.resultq.close()
            raise e

        # Set up a poison pill
        poisonpill = False

        # Loop to listen for jobs
        while True:
            jobs = []
            for _ in range(self.workerconfig['batchsize']):
                # Fetch from queue
                try:
                    jobs.append(self.jobq.get(block=False))
                except q.Empty:
                    poisonpill = True
                    break

            self.print_("[+] Fetched {} jobs from JobQ. Fetching corresponding tensor and augmenting...".format(len(jobs)))
            try:
                # Fetch tensor
                inp = self.fetchtensor(jobs=jobs)
                self.print_("[+] Fetch input batch of shape {}.".format(inp.shape))

                # Generate random augmentation function
                self.genrandaug(batchshape=inp.shape)

                # Augment inpunt
                auginp = self.augfunc(inp)
                self.print_("[+] Augmented input batch. The shape now is {}.".format(auginp.shape))

            except Exception as e:
                self.print_(
                    "[-] Exception raised while fetching tensor and/or applying data augmentation. "
                    "The error message follows: {}".format(e.message))
                # Send poison pill and call it a day
                self.resultq.put(None)
                self.resultq.close()
                raise e

            self.print_("[+] Inferring...")
            try:
                # Process
                out = network.classifier(auginp)
                self.print_("[+] Output from the network is of shape {}.".format(out.shape))
            except Exception as e:
                self.print_("[-] Exception raised while running inference. The error message follows: {}".format(e.message))
                # Send poison pill and call it a day
                self.resultq.put(None)
                self.resultq.close()
                raise e

            self.print_("[+] Deaugmenting...")
            try:
                # Deaugment output
                deaugout = self.deaugfunc(out)
                self.print_("[+] Deaugmented network output. "
                            "The shape of the deaugmented batch is {}.".format(deaugout.shape))

            except Exception as e:
                self.print_(
                    "[-] Exception raised while deaugmenting processed data. "
                    "The error message follows: {}".format(e.message))
                # Send poison pill and call it a day
                self.resultq.put(None)
                self.resultq.close()
                raise e

            self.print_("[+] Writing output to ResultQ.")
            # Write results to the results queue
            for outimg, job in zip(deaugout, jobs):
                self.resultq.put({'dataset': job['dataset'], 'slice': job['slice'], 'payload': outimg})

            # Check for death wish
            if poisonpill:
                self.print_("[-] Poison pill found, shutting down process.")
                # Set up suicide pact
                self.resultq.put(None)
                self.resultq.close()
                break


class supervisor(object):
    def __init__(self, superconfig, verbose=True):
        # Meta
        self.superconfig = superconfig
        self.verbose = verbose
        self.workerconfigs = None
        self.workerlist = None

        # Configure logger
        self.logger = None
        self.configurelogger()

        # Container for extra baggage
        self.baggage = {}

        # Initialize job and result q's
        self.jobq = mp.Queue()
        self.resultq = mp.Queue()

        # Initialize and populate volume storage
        self.datasets = None
        self.load(pad=True)

        # Populate JobQ
        self.jobcenter()

        # Build workerconfig
        self.buildworkerconfigs()

        # Prepare volumes required by the writer method to do its job
        self.prepwriter()

        # Set up result preprocessor
        self.resultpreprocessor = self._cremi_resultpreprocessor

    def print_(self, msg):
        if self.verbose:
            print("Supervisor {}: {}".format(os.getpid(), msg))

        if self.logger is not None:
            self.logger("Supervisor {}: {}".format(os.getpid(), msg))

    def configurelogger(self):
        logfile = self.superconfig['logfile'] if 'logfile' in self.superconfig.keys() else None
        if logfile is not None:
            self.logger = logger(logfile)

    def __call__(self):
        self.run()

    def pad(self, volume):
        """
        Pad volume such that its X and Y shape is a multiple of a given number (to be provided in the
        config as 'padmultiple').
        """

        volshape = volume.shape
        Zshape, Yshape, Xshape = volshape

        # X and Y are to be padded to a multiple of 16
        padmultiple = self.superconfig['padmultiple']

        # Get 2 * pad
        Yfullpad = padmultiple - (Yshape % padmultiple)
        Xfullpad = padmultiple - (Xshape % padmultiple)

        # Make sure the volume is paddable
        assert (Xfullpad % 2, Yfullpad % 2) == (0, 0), "Volume is not paddable."

        # Get actual padding
        Xpad = Xfullpad/2
        Ypad = Yfullpad/2
        Zpad = (self.superconfig['numzslices'] - 1)/2

        # Pad baby
        pvolume = np.pad(volume, pad_width=((Zpad, Zpad), (Ypad, Ypad), (Xpad, Xpad)), mode='reflect')

        # Write out
        return {'padded_volume': pvolume, 'padconfig': (Zpad, Ypad, Xpad), 'volshape': pvolume.shape}

    def getslicelist(self, volshape, padconfig):
        """
        Given the shape of a volume `volshape` and how it was padded `padconfig`, generate a list of slice tuples.
        The entire volume is processed if this list is exhausted.
        """
        # Allowed are the cases with 2x and without downsampling.
        assert self.superconfig['ds'] in [1, 2]

        # Preallocate a list of slices
        slicelist = []
        # Get padding configuration
        Zpad, Ypad, Xpad = padconfig

        for planenum in range(Zpad, volshape[0]-Zpad):
            # Get Z slice
            Zsl = slice(planenum - Zpad, planenum + Zpad + 1)

            # Get X, Y slices
            if self.superconfig['ds'] == 1:
                # Get the entire slice
                Ysl = slice(0, None)
                Xsl = slice(0, None)

                # Add slices numfold times
                for _ in range(self.superconfig['numfolds']):
                    slicelist.append((Zsl, Ysl, Xsl))

            else:

                # Get downsampled slices - all 4 of 'em
                for starty, startx in product((0, 1), (0, 1)):
                    Ysl = slice(starty, None, 2)
                    Xsl = slice(startx, None, 2)

                    # Add slices numfold times
                    for _ in range(self.superconfig['numfolds']):
                        slicelist.append((Zsl, Ysl, Xsl))

        # Return slicelist
        return slicelist

    def load(self, pad=True, normalize=True):
        """Load datasets to RAM"""
        from Antipasti.netdatautils import fromh5

        # Load from H5
        datasets = {dset: fromh5(pathsy(self.superconfig['datapaths'][dset]),
                                 self.superconfig['h5paths'][dset]).transpose(2, 1, 0)
                    for dset in self.superconfig['datapaths'].keys()}

        self.print_("[+] Loaded volumes from HDF5.")

        # Pad datasets if required
        if pad:
            datasetsconinfo = {dset: self.pad(dvol) for dset, dvol in datasets.items()}
            self.datasets = {dset: dci['padded_volume'] for dset, dci in datasetsconinfo.items()}
            self.baggage["padconfig"] = {dset: dci['padconfig'] for dset, dci in datasetsconinfo.items()}
            self.baggage["volshapes"] = {dset: dci['volshape'] for dset, dci in datasetsconinfo.items()}
            self.print_("[+] Padded volumes.")
        else:
            self.print_("[-] Not padding volumes.")
            self.datasets = datasets

        if normalize:

            def normalizevolume(vol):
                vol = (vol - vol.mean())/vol.std()
                return vol

            self.print_("[+] Normalizing volumes...")
            # Normalzie all volumes
            self.datasets = {dset: normalizevolume(dvol) for dset, dvol in self.datasets.items()}
        else:
            self.print_("[-] Not normalizing volumes...")

    def jobcenter(self):
        """Build a list of jobs to be processed by workers."""

        self.print_("[+] Populating JobQ...")
        # Job counter
        self.baggage['numjobs'] = 0

        for dset in self.datasets.keys():
            # Get slicelist for this dataset
            slicelist = self.getslicelist(self.baggage['volshapes'][dset], self.baggage['padconfig'][dset])

            # Loop over slicelist and add job to joblist
            for sl in slicelist:
                # Increment job counter
                self.baggage['numjobs'] += 1
                # Queue job
                self.jobq.put({'dataset': dset, 'slice': sl}, block=False)

    def buildworkerconfigs(self):
        """Build config dicts for all workers."""
        workerconfigs = []

        for workernum in range(self.superconfig['numworkers']):
            # Check if a device list is specified
            device = self.superconfig['devices'][workernum] \
                if 'devices' in self.superconfig else 'gpu{}'.format(workernum)

            # Check if a PID needs to be waited for
            if 'waitforpid' in self.superconfig.keys():
                try:
                    waitpid = self.superconfig['waitforpid'][workernum]
                    wait = [('waitforpid', waitpid)] if waitpid is not None else []

                    # check with user if config correct
                    if waitpid is not None:
                        configok = raw_input("Process will wait for PID {} "
                                             "to compute on {}. Okay? (y/n)".format(waitpid, device))
                    else:
                        configok = raw_input("Process will not wait for anyone "
                                             "to compute on {}. Okay? (y/n)".format(device))

                    if configok == 'y':
                        pass
                    elif configok == 'n':
                        raise RuntimeError("Canceled.")
                    else:
                        raise NotImplementedError("Answer must be y or n.")

                except IndexError:
                    wait = []
            else:
                wait = []

            workerconfig = dict(self.superconfig.items() +
                                [('device', device), ('datasets', self.datasets)] + wait)
            workerconfigs.append(workerconfig)


        self.workerconfigs = workerconfigs

    def prepwriter(self):
        """Prepare writer. This class initializes the write and normalization volumes, which are written to baggage."""
        # Get volume shapes
        self.baggage['outvols'] = {dset: np.zeros(shape=dshape) for dset, dshape in self.baggage['volshapes'].items()}
        self.baggage['normvols'] = {dset: np.zeros(shape=dshape) for dset, dshape in self.baggage['volshapes'].items()}

    def writer(self, result):
        # Preprocess results
        result = self.resultpreprocessor(result)
        # Write payload
        self.baggage['outvols'][result['dataset']][result['slice']] += result['payload']
        # Increment normalization volume
        self.baggage['normvols'][result['dataset']][result['slice']] += 1.

    def mayday(self):
        self.finish()

    def finish(self):
        from Antipasti.netdatautils import toh5
        # Normalize volumes
        for dset, outvol in self.baggage['outvols'].items():
            # Get normalization volume
            normvol = self.baggage['normvols'][dset]
            # Get rid of zeros in normvol
            normvol[normvol == 0.] = 1.
            # Average
            writevol = outvol/normvol
            # Write to file
            toh5(writevol, pathsy(self.superconfig['writepaths'][dset]))
            self.print_("[+] Wrote dataset {} to {}.".format(dset, pathsy(self.superconfig['writepaths'][dset])))
            # Done.

    # Populate worker list
    def hire(self):
        workerlist = []
        for workerconfig in self.workerconfigs:
            workerlist.append(worker(workerconfig, self.jobq, self.resultq))
        self.workerlist = workerlist
        pass

    def run(self):
        self.print_("[+] Hiring workers...")
        # Hire workers if required
        if self.workerlist is None:
            self.hire()

        self.print_("[+] Starting workers...")
        # Start all workers
        for wrkr in self.workerlist:
            wrkr.start()

        # Listen for results
        # Set up a poison pill counter
        ppcount = 0
        # Every worker must send the supervisor one poison pill
        maxppcount = len(self.workerlist)

        self.print_("[+] Workers have started working. Listening for results.")

        # Count number of results written.
        self.baggage['numjobsdone'] = 0

        while True:
            result = self.resultq.get(block=True)

            self.print_("[+] Fetched results.")

            if result is None:
                ppcount += 1
                self.print_("[+] Posion pill {} of {} found.".format(ppcount, maxppcount))
            else:
                self.writer(result)
                self.baggage['numjobsdone'] += 1
                self.print_("[+] Wrote result {} of {} to volume."
                            "Have {} of the {} required poison pills.".format(self.baggage['numjobsdone'],
                                                                              self.baggage['numjobs'],
                                                                              ppcount, maxppcount))

            # Check if supervisor has all its poison pills
            if ppcount == maxppcount:
                self.print_("[+] Breaking out from listen loop.")
                break

            if self.baggage['numjobsdone'] >= self.baggage['numjobs']:
                self.print_("[-] Something went wrong with the suicide pact. This is exit mechanism a failsafe.")
                break

        self.print_("[+] Writing out to file...")
        # Write out to file
        self.finish()

        self.print_("[+] Cleaning up: joining workers...")
        # Join all workers
        for wrkr in self.workerlist:
            wrkr.join()

        self.print_("[+] Done.")
        raise SystemError

    def _cremi_resultpreprocessor(self, result):
        # Get halfwindow
        halfwindow = self.baggage['padconfig'][result['dataset']][0]
        # Pick central frame in data
        result['payload'] = result['payload'][halfwindow:halfwindow+1, ...]
        result['slice'] = (slice(result['slice'][0].start + halfwindow,
                                 result['slice'][0].start + halfwindow + 1),
                           result['slice'][1],
                           result['slice'][2])
        return result


class logger(object):
    def __init__(self, logfile):
        """
        A very basic text based logger. The file is open within a context manager, so it's safe against external
        interruptions.

        :type logfile: str
        :param logfile: Path to the logfile.
        """
        self.logfile = pathsy(logfile)

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

if __name__ == '__main__':
    parsey = argparse.ArgumentParser()
    parsey.add_argument('inferconfigset', help="Inference configuration.")
    args = parsey.parse_args()
    inferconfig = pathsy(args.inferconfigset)

    # Read worker config
    with open(inferconfig) as f: superconfig = yaml.load(f)
    # Set up supervisor
    sprvsr = supervisor(superconfig)
    # Run supervisor
    sprvsr.run()
