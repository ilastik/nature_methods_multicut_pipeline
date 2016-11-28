
__author__ = 'nasimrahaman'

""" Module to handle data handling. """

import numpy as np
import prepkit as pk
import itertools as it
import pickle as pkl

# Class to convert any given generator to a Antipasti datafeeder (endowed with a restartgenerator() method)
class feeder(object):
    def __init__(self, generator, genargs=None, genkwargs=None, preptrain=None):
        """
        Convert a given generator to an Antipasti data feeder (endowed with a restartgenerator and batchstream method).

        :type generator: generator
        :param generator: Generator to be converted to a feeder.

        :type genargs: list
        :param genargs: List of arguments generator may take.

        :type genkwargs: dict
        :param genkwargs: Dictionary of keyword arguments generator may take.

        :type preptrain: prepkit.preptrain
        :param preptrain: Train of preprocessing functions.

        """

        assert callable(generator), "Generator must be callable."

        # Meta + Defaults
        self.generator = generator
        self.genargs = genargs if genargs is not None else []
        self.genkwargs = genkwargs if genkwargs is not None else {}
        self.preptrain = pk.preptrain([]) if preptrain is None else preptrain

        self.iterator = None
        self.restartgenerator()

    def restartgenerator(self):
        self.iterator = self.generator(*self.genargs, **self.genkwargs)

    def batchstream(self):
        # Stream batches after having applied preptrain
        for batch in self.iterator:
            yield self.preptrain(batch)

    def __iter__(self):
        return self

    def next(self):
        return self.batchstream().next()

# Class to zip multiple generators
class feederzip(object):
    """
    Zip multiple generators (with or without a restartgenerator method)
    """
    def __init__(self, gens, preptrain=None):
        """
        :type preptrain: prepkit.preptrain
        :param preptrain: Train of preprocessing functions

        :type gens: list of generators
        :param gens: List of generators to be zipped.

        :return:
        """
        # Meta
        self.gens = gens
        self.preptrain = preptrain if preptrain is not None else pk.preptrain([])

    def batchstream(self):
        # Fetch from all generators and yield
        while True:
            batchlist = [gen.next() for gen in self.gens]
            yield self.preptrain(batchlist)

    def restartgenerator(self):
        # Restart generator where possible
        for gen in self.gens:
            if hasattr(gen, "restartgenerator"):
                gen.restartgenerator()

    # To use feederzip as an iterator
    def __iter__(self):
        return self

    # Next method to mirror batchstream
    def next(self):
        return self.batchstream().next()