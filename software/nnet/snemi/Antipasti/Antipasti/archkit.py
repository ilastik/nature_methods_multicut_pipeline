__author__ = "Nasim Rahaman"

__doc__ = """ This module contains architectural layers for building networks with general topologies."""

# Imports
import theano as th
import theano.tensor as T
import numpy as np

import copy

from Antipasti.netkit import layer
import Antipasti.netutils as netutils
import pykit as pyk

import lasagne as las


# Time average layer
class timeaveragelayer(layer):

    """Layer to time average sequential inputs."""
    def __init__(self, keepdims=False, inpshape=None):
        """
        :type keepdims: bool
        :param keepdims: Whether to keep the T dimension or to squeeze it away.

        :type inpshape: list or tuple
        :param inpshape: Shape of the input tensor
        """
        super(timeaveragelayer, self).__init__()

        # Meta
        self.keepdims = keepdims

        # Input must be 5D sequential, i.e.
        self.dim = 2
        self.inpdim = 5
        self.allowsequences = True
        self.issequence = True

        # Shape inference
        self.inpshape = list(inpshape) if inpshape is not None else [None, ] * self.inpdim

        self.layerinfo = "[Keep Dimensions: {}]".format(self.keepdims)

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        self.y = T.tensor('floatX', [False, ] * (self.inpdim - (0 if not self.keepdims else 1)),
                          name='y:' + str(id(self)))
        self.xr = T.tensor('floatX', [False, ] * self.inpdim, name='xr:' + str(id(self)))

    def feedforward(self, inp=None):
        # Parse
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Input is 5D sequential, i.e. we average over the T axis
        self.y = T.unbroadcast(T.mean(inp, axis=1, keepdims=self.keepdims), 1)

        # Return
        return self.y

    def inferoutshape(self, inpshape=None, checkinput=True):
        # Parse
        if inpshape is None:
            inpshape = self.inpshape

        # Return inpshape unchanged if encoder is not active,
        if not self.encoderactive:
            return inpshape

        if checkinput:
            assert len(inpshape) == 5, "Input must be 5D sequential."

        # The actual shape inference itself must be somewhat robust
        outshape = inpshape[0:1] + ([1] if self.keepdims else []) + inpshape[2:]

        return outshape


class temporalizelayer(layer):
    """Layer to add a temporal axis to the input."""
    def __init__(self, inpshape=None):
        super(temporalizelayer, self).__init__()

        # Input must be non-sequential
        self.dim = 2
        self.inpdim = 4
        self.issequence = False
        self.allowsequences = False

        # Shape inference
        self.inpshape = list(inpshape) if inpshape is not None else [None, ] * self.inpdim

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        self.y = T.tensor('floatX', [False, ] * (self.inpdim + 1), name='y:' + str(id(self)))
        self.xr = T.tensor('floatX', [False, ] * self.inpdim, name='xr:' + str(id(self)))

    def feedforward(self, inp=None):
        # Parse
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Add in a new axis and unbroadcast it
        out = T.unbroadcast(inp.dimshuffle(0, 'x', 1, 2, 3), 1)

        # Return
        return out

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        # Check input shape
        if checkinput:
            assert len(inpshape) == 4

        # Shape inference
        outshape = inpshape[0:1] + [1] + inpshape[1:]

        # Return
        return outshape


class idlayer(layer):
    """Layer to implement skip connections."""
    def __init__(self, dim=2, issequence=False, inpshape=None):
        """
        :type dim: int
        :param dim: Dimension of the data (= 2 for images, = 3 for sequential or 3D)

        :type inpshape: list or tuple
        :param inpshape: Input shape
        :return:
        """
        super(idlayer, self).__init__()

        # Parse data dimensionality
        assert not (dim is None and inpshape is None), "Data dimension can not be parsed. Provide dim or inpshape."

        # Meta
        self.dim = dim if dim is not None else {4: 2, 5: 3}[len(inpshape)]
        self.allowsequences = True
        self.issequence = self.dim == 2 and len(self.inpshape) == 5 if issequence is None else issequence
        self.inpdim = len(inpshape) if inpshape is not None else 5 if self.issequence else {2: 4, 3: 5}[dim]

        # Shape inference
        self.inpshape = [None, ] * self.inpdim if inpshape is None else list(inpshape)

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        self.y = T.tensor('floatX', [False, ] * self.inpdim, name='y:' + str(id(self)))

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # kek
        self.y = inp

        return self.y

    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        return inpshape


class splitlayer(layer):
    """Layer to split (by channel axis) one input tensor to multiple output tensors."""
    def __init__(self, splits, dim=None, issequence=None, inpshape=None):
        """
        :type splits: list or int
        :param splits: Index of the split (along the channel axis). E.g. split = 3 would result in the input tensor
                       split as: [inp[:, 0:3, ...], inp[:, 3:, ...]] for 2D inputs.

        :type issequence: bool
        :param issequence: Whether input is a sequence

        :type inpshape: list or tuple
        :param inpshape: Input shape
        :return:
        """

        super(splitlayer, self).__init__()

        # Parse
        dim = 2 if issequence else dim
        assert not (dim is None and inpshape is None), "Data dimension can not be parsed. Provide dim or inpshape."

        # Meta
        self.dim = dim if dim is not None else {4: 2, 5: 3}[len(inpshape)]
        self.allowsequences = True
        self.issequence = self.dim == 2 and len(self.inpshape) == 5 if issequence is None else issequence
        self.inpdim = len(inpshape) if inpshape is not None else 5 if self.issequence else {2: 4, 3: 5}[dim]
        self.dim = 2 if self.issequence else self.dim   # Correct dim if necessary

        self.splits = pyk.obj2list(splits)
        self.numsplits = len(self.splits) + 1

        # More meta for layertrainyard
        self.numinp = 1
        self.numout = self.numsplits

        # Shape inference
        self.inpshape = [None, ] * self.inpdim if inpshape is None else list(inpshape)

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        self.y = [T.tensor('floatX', [False, ] * self.inpdim, name='y{}:'.format(splitnum) + str(id(self)))
                  for splitnum in range(self.numsplits)]

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Split
        out = []
        cursor = 0
        for splitloc in self.splits + [None]:
            if self.inpdim == 4:
                out.append(inp[:, cursor:splitloc, :, :])
            elif self.inpdim == 5:
                out.append(inp[:, :, cursor:splitloc, :, :])
            cursor = cursor + splitloc if splitloc is not None else None

        # Return
        self.y = out
        return self.y

    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        if checkinput:
            assert len(pyk.obj2list(inpshape[0])) == 1, "Input shape must be a list of ints " \
                                                        "(split layer takes in 1 input)"

        # Recall that outshape must be a list of lists
        outshape = []
        shape = inpshape

        indsplits = [0] + self.splits + [inpshape[-1]]

        for n in range(len(indsplits) - 1):
            # Copy shape (this is necessary, because python sets by reference)
            shape = copy.copy(shape)
            # Update channel size for all outputs
            shape[(1 if self.inpdim == 4 else 2)] = indsplits[n + 1] - indsplits[n] \
                if indsplits[n + 1] is not None else None
            outshape.append(shape)

        # Return
        return outshape


class mergelayer(layer):
    """Layer to merge multiple inputs to one output tensor."""
    def __init__(self, numinp=None, dim=2, issequence=False, inpshape=None):
        """
        :type numinp: int
        :param numinp: Number of inputs

        :type dim: list or int
        :param dim: List (or int) of data dimensions of the input

        :type issequence: list or bool
        :param issequence: Whether the inputs are sequences.

        :type inpshape: list or tuple
        :param inpshape: Input shape
        :return:
        """
        super(mergelayer, self).__init__()

        # Convenience parse
        dim = [dim, ] * numinp if isinstance(dim, int) and numinp is not None else dim
        issequence = [issequence, ] * numinp if isinstance(issequence, bool) and numinp is not None else issequence

        # Parse inpshape if numinp, dim, issequence is given
        if numinp is not None and dim is not None and inpshape is None:
            if 3 in dim:
                # issequence doesn't matter, inpshape can still be filled
                inpshape = [[None, ] * 5 for _ in dim]
            else:
                # dim is 2, but input might still be sequential
                if issequence is None:
                    pass
                else:
                    if issequence:
                        # issequence is false and dim = 2, ergo 2D data
                        inpshape = [[None, ] * 4 for _ in dim]
                    else:
                        # issequence is true, ergo 2D sequential data
                        inpshape = [[None, ] * 5 for _ in dim]

        # Check
        assert not (numinp is None and inpshape is None), "Number of inputs could not be parsed. Provide numinp " \
                                                          "or inpshape."

        assert not (all([idim is None for idim in pyk.obj2list(dim)]) and inpshape is None), \
            "Data dimension could not be parsed. Please provide dim or inpshape."

        assert isinstance(dim, list) if dim is not None else True, "Dim must be a list for multiple inputs."

        assert all([isinstance(ishp, list) for ishp in inpshape]) if inpshape is not None else True, \
            "mergelayer expects multiple inputs, but inpshape doesn't have the correct signature."

        # Meta
        self.numinp = numinp if numinp is not None else len(inpshape)
        self.numout = 1
        self.dim = dim if dim is not None else [{4: 2, 5: 2}[len(ishp)] for ishp in inpshape]
        self.allowsequences = True
        self.issequence = [idim == 2 and len(ishp) == 5 for idim, ishp in zip(self.dim, inpshape)] \
            if issequence is None else issequence
        self.inpdim = [len(ishp) if ishp is not None else 5 if iisseq else {2: 4, 3: 5}[idim]
                       for ishp, iisseq, idim in zip(inpshape, self.issequence, self.dim)]

        # Check if inpdim is consistent
        assert all([indim == self.inpdim[0] for indim in self.inpdim]), "Can't concatenate 2D with 3D inputs."

        # Shape inference
        self.inpshape = list(inpshape) if inpshape is not None else [[None, ] * indim for indim in self.inpdim]

        # Containers for input and output
        self.x = [T.tensor('floatX', [False, ] * indim, name='x{}:'.format(inpnum) + str(id(self)))
                  for inpnum, indim in enumerate(self.inpdim)]
        self.y = T.tensor('floatX', [False, ] * self.inpdim[0], name='y:' + str(id(self)))

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Find axis along which to concatenate
        cataxis = 1 if self.inpdim[0] == 4 else 2

        # Concatenate
        out = T.concatenate(tuple(inp), axis=cataxis)

        # Return
        self.y = out
        return self.y

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        # Find channel axis
        chaxis = 1 if self.inpdim[0] == 4 else 2

        if checkinput:
            # Make sure correct number of inputs
            sum([isinstance(ishp, list) for ishp in inpshape]) == self.numinp, "Inpshape must be a list of input " \
                                                                               "shapes of {} layer " \
                                                                               "inputs.".format(self.numinp)
            # Make sure input shapes are consistent
            assert all([ishp[:chaxis] + ishp[(chaxis+1):] == inpshape[0][:chaxis] + inpshape[0][(chaxis+1):]
                        for ishp in inpshape]), "All inputs must have the same size in all but the channel axis."

        outshape = inpshape[0][0:chaxis] + \
                   [sum([ishp[chaxis] for ishp in inpshape])
                    if None not in [ishp[chaxis] for ishp in inpshape] else None] + \
                   inpshape[0][(chaxis+1):]

        return outshape


class replicatelayer(layer):
    """Layer to replicate input to multiple outputs"""
    def __init__(self, numreplicate, dim=2, issequence=False, inpshape=None):
        """
        :type numreplicate: int
        :param numreplicate: Number of times to replicate

        :type dim: int
        :param dim: Dimensionality of input data

        :type inpshape: list
        :param inpshape: Input shape
        :return:
        """
        super(replicatelayer, self).__init__()

        assert not (dim is None and inpshape is None), "Data dimension can not be parsed. Provide dim or inpshape."

        # Meta
        self.dim = dim if dim is not None else {4: 2, 5: 3}[len(inpshape)]
        self.allowsequences = True
        self.issequence = self.dim == 2 and len(self.inpshape) == 5 if issequence is None else issequence
        self.inpdim = len(inpshape) if inpshape is not None else 5 if self.issequence else {2: 4, 3: 5}[dim]
        self.dim = 2 if self.issequence else self.dim   # Correct dim if necessary

        self.numcopies = numreplicate

        # More meta for layertrainyard
        self.numinp = 1
        self.numout = self.numcopies

        # Shape inference
        self.inpshape = [None, ] * self.inpdim if inpshape is None else list(inpshape)

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        self.y = [T.tensor('floatX', [False, ] * self.inpdim, name='y{}:'.format(splitnum) + str(id(self)))
                  for splitnum in range(self.numcopies)]

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Replicate inp
        out = [inp, ] * self.numcopies

        # Return
        self.y = out
        return self.y

    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        return [inpshape, ] * self.numcopies


# Class for circuit connections
class circuitlayer(layer):
    """Layer to merge and replicate multiple inputs to multiple outputs."""
    def __init__(self, connections, merge=True, dim=2, issequence=False, inpshape=None):
        """
        :type connections: list
        :param connections: List of connections. E.g.: [[0, 1], 1, [0, 1, 2]] would merge inputs 0 & 1 and write to
                            first output slot, 1 to second output slot and 0, 1 & 2 merged to the third output slot.
        :param dim:
        :param issequence:
        :param inpshape:
        :return:
        """
        super(circuitlayer, self).__init__()

        # Compute the number of i/o slots
        self.numinp = len(pyk.unique(list(pyk.flatten(connections))))
        self.numout = len(connections)
        self.merge = bool(merge)

        # TODO Fix this
        assert self.numinp != 1, "Circuit layer must have atleast 2 inputs. Consider using a replicatelayer."

        # Parse
        dim = [dim, ] * self.numinp if isinstance(dim, int) and self.numinp is not None else dim
        issequence = [issequence, ] * self.numinp if isinstance(issequence, bool) and self.numinp is not None \
            else issequence

        # Parse inpshape if numinp, dim, issequence is given
        if self.numinp is not None and dim is not None and inpshape is None:
            if 3 in dim:
                # issequence doesn't matter, inpshape can still be filled
                inpshape = [[None, ] * 5 for _ in dim]
            else:
                # dim is 2, but input might still be sequential
                if issequence is None:
                    pass
                else:
                    if issequence:
                        # issequence is false and dim = 2, ergo 2D data
                        inpshape = [[None, ] * 4 for _ in dim]
                    else:
                        # issequence is true, ergo 2D sequential data
                        inpshape = [[None, ] * 5 for _ in dim]

        # Meta
        self.connections = connections
        self.dim = dim if dim is not None else [{4: 2, 5: 2}[len(ishp)] for ishp in inpshape]
        self.allowsequences = True
        self.issequence = [idim == 2 and len(ishp) == 5 for idim, ishp in zip(self.dim, inpshape)] \
            if issequence is None else issequence
        self.inpdim = [len(ishp) if ishp is not None else 5 if iisseq else {2: 4, 3: 5}[idim]
                       for ishp, iisseq, idim in zip(inpshape, self.issequence, self.dim)]

        # Check if inpdim is consistent
        assert all([indim == self.inpdim[0] for indim in self.inpdim]), "Can't concatenate 2D with 3D inputs."

        # Make merge layers
        self.mergelayers = [mergelayer(numinp=len(conn),
                                       dim=[self.dim[node] for node in conn],
                                       issequence=[self.issequence[node] for node in conn],
                                       inpshape=[inpshape[node] for node in conn])
                            if pyk.smartlen(conn) != 1 else None for conn in self.connections]

        # Shape inference
        self.inpshape = list(inpshape) if inpshape is not None else [[None, ] * indim for indim in self.inpdim]

        # Containers for input and output
        self.x = pyk.delist([T.tensor('floatX', [False, ] * indim, name='x{}:'.format(inpnum) + str(id(self)))
                             for inpnum, indim in enumerate(self.inpdim)])
        self.y = pyk.delist([T.tensor('floatX', [False, ] * self.inpdim[pyk.obj2list(self.connections[outnum])[0]],
                                      name='y:' + str(id(self)))
                            for outnum in range(self.numout)])

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        # Buffer for outshape
        outshape = []
        for connum, conn in enumerate(self.connections):
            if self.mergelayers[connum] is None:
                # conn is the index of an element in the input list. Fetch its shape:
                outshape.append(self.inpshape[pyk.delist(pyk.obj2list(conn))])
            else:
                if self.merge:
                    # Fetch merge layer's inpshape
                    self.mergelayers[connum].inpshape = [inpshape[node] for node in conn]
                    # Append outshape
                    outshape.append(self.mergelayers[connum].outshape)
                else:
                    outshape.append([inpshape[node] for node in conn])

        return pyk.delist(outshape)

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Loop over connections, merge and append to a buffer
        out = []
        for connum, conn in enumerate(self.connections):
            if pyk.smartlen(conn) == 1:
                out.append(inp[pyk.delist(pyk.obj2list(conn))])
            else:
                if self.merge:
                    out.append(self.mergelayers[connum].feedforward(inp=[inp[node] for node in conn]))
                else:
                    out.append([inp[node] for node in conn])

        self.y = out
        return self.y


class addlayer(layer):
    """Layer to add multiple inputs to one output tensor."""
    def __init__(self, numinp=None, dim=None, issequence=None, inpshape=None):
        """
        :type numinp: int
        :param numinp: Number of inputs

        :type dim: list or int
        :param dim: List (or int) of data dimensions of the input

        :type issequence: list or bool
        :param issequence: Whether inputs are sequences

        :type inpshape: list or tuple
        :param inpshape: Input shape
        :return:
        """
        super(addlayer, self).__init__()

        # Convenience parse
        dim = [dim, ] * numinp if isinstance(dim, int) and numinp is not None else dim
        issequence = [issequence, ] * numinp if isinstance(issequence, bool) and numinp is not None else issequence

        # Parse inpshape if numinp, dim, issequence is given
        if numinp is not None and dim is not None and inpshape is None:
            if 3 in dim:
                # issequence doesn't matter, inpshape can still be filled
                inpshape = [[None, ] * 5 for _ in dim]
            else:
                # dim is 2, but input might still be sequential
                if issequence is None:
                    pass
                else:
                    if issequence:
                        # issequence is false and dim = 2, ergo 2D data
                        inpshape = [[None, ] * 4 for _ in dim]
                    else:
                        # issequence is true, ergo 2D sequential data
                        inpshape = [[None, ] * 5 for _ in dim]

        # Check
        assert not (numinp is None and inpshape is None), "Number of inputs could not be parsed. Provide numinp " \
                                                          "or inpshape."

        assert not (all([idim is None for idim in pyk.obj2list(dim)]) and inpshape is None), \
            "Data dimension could not be parsed. Please provide dim or inpshape."

        assert isinstance(dim, list) if dim is not None else True, "Dim must be a list for multiple inputs."

        assert all([isinstance(ishp, list) for ishp in inpshape]) if inpshape is not None else True, \
            "mergelayer expects multiple inputs, but inpshape doesn't have the correct signature."

        # Meta
        self.numinp = numinp if numinp is not None else len(inpshape)
        self.numout = 1
        self.dim = dim if dim is not None else [{4: 2, 5: 2}[len(ishp)] for ishp in inpshape]
        self.allowsequences = True
        self.issequence = [idim == 2 and len(ishp) == 5 for idim, ishp in zip(self.dim, inpshape)] \
            if issequence is None else issequence
        self.inpdim = [len(ishp) if ishp is not None else 5 if iisseq else {2: 4, 3: 5}[idim]
                       for ishp, iisseq, idim in zip(inpshape, self.issequence, self.dim)]

        # Check if inpdim is consistent
        assert all([indim == self.inpdim[0] for indim in self.inpdim]), "Can't concatenate 2D with 3D inputs."

        # Shape inference
        self.inpshape = list(inpshape) if inpshape is not None else [[None, ] * indim for indim in self.inpdim]

        # Containers for input and output
        self.x = [T.tensor('floatX', [False, ] * indim, name='x{}:'.format(inpnum) + str(id(self)))
                  for inpnum, indim in enumerate(self.inpdim)]
        self.y = T.tensor('floatX', [False, ] * self.inpdim[0], name='y:' + str(id(self)))

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Concatenate
        out = sum(inp)

        # Return
        self.y = out
        return self.y

    def inferoutshape(self, inpshape=None, checkinput=True):
        if inpshape is None:
            inpshape = self.inpshape

        if checkinput:
            # Make sure correct number of inputs
            sum([isinstance(ishp, list) for ishp in inpshape]) == self.numinp, "Inpshape must be a list of input " \
                                                                               "shapes of {} layer " \
                                                                               "inputs.".format(self.numinp)
            assert all([ishp == inpshape[0] for ishp in inpshape]), "All inputs must have the same shape."

        outshape = inpshape[0]

        return outshape


class activationlayer(layer):
    """ Layer to apply an element-wise function on the input. """
    def __init__(self, activation, dim=2, issequence=False, inpshape=None):
        """
        :type activation: callable or dict
        :param activation: Activation function (any element-wise symbolic function)

        :type dim: int
        :param dim: Dimensionality of the input data

        :type issequence: bool
        :param issequence: Whether the input is a sequence

        :type inpshape: list
        :param inpshape: Input shape
        """

        super(activationlayer, self).__init__()

        # Parse data dimensionality
        assert not (dim is None and inpshape is None), "Data dimension can not be parsed. Provide dim or inpshape."

        # Meta
        self.dim = dim if dim is not None else {4: 2, 5: 3}[len(inpshape)]
        self.allowsequences = True
        self.issequence = self.dim == 2 and len(inpshape) == 5 if issequence is None else issequence
        self.inpdim = len(inpshape) if inpshape is not None else 5 if self.issequence else {2: 4, 3: 5}[dim]

        # Parse activation
        if isinstance(activation, dict):
            self.activation = activation["function"]
            self._params = pyk.obj2list(activation["trainables"]) if "trainables" in activation.keys() else []
            self._cparams = pyk.obj2list(activation["ctrainables"]) if "ctrainables" in activation.keys() else \
                [netutils.getshared(like=trainable, value=1.) for trainable in self.params]
            # Name shared variables in params / cparams
            for n, (param, cparam) in enumerate(zip(self.params, self.cparams)):
                param.name += "-trainable{}:".format(n) + str(id(self))
                cparam.name += "-ctrainable{}:".format(n) + str(id(self))
        else:
            self.activation = activation

        # Shape inference
        self.inpshape = [None, ] * self.inpdim if inpshape is None else list(inpshape)

        # Containers for input and output
        self.x = T.tensor('floatX', [False, ] * self.inpdim, name='x:' + str(id(self)))
        self.y = T.tensor('floatX', [False, ] * self.inpdim, name='y:' + str(id(self)))

    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        self.y = self.activation(inp)

        return self.y

    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        return inpshape


class functionlayer(layer):
    def __init__(self, func, shapefunc=None, funcargs=None, funckwargs=None, shapefuncargs=None, shapefunckwargs=None,
                 numinp=1, numout=1, dim=None, issequence=None, inpshape=None):
        """
        Layer to apply any given function to input(s). The function may require multiple inputs and
        return multiple outputs. The function may change the shape of the tensor, but then a `shapefunc`
        must be provided for automatic shape inference.

        :type func: callable
        :param func: Function to apply to input.

        :type shapefunc: callable
        :param shapefunc: Function to compute the output shape given input shape.

        :type funcargs: tuple or list
        :param funcargs: Arguments for the function (`func`)

        :type funckwargs: dict
        :param funckwargs: Keyword arguments for the function (`func`)

        :type shapefuncargs: tuple or list
        :param shapefuncargs: Arguments for the shape function (`shapefunc`)

        :type shapefunckwargs: dict
        :param shapefunckwargs: Keyword arguments for the shape function (`shapefunc`)

        :type numinp: int
        :param numinp: Number of inputs the function `func` takes.

        :type numout: int
        :param numout: Number of outputs the function `func` returns.

        :type dim: int or list of int
        :param dim: Dimensionality of the input data. Defaults to 2 when omitted.

        :type issequence: bool or list of bool
        :param issequence: Whether the input(s) is sequential.
        """
        super(functionlayer, self).__init__()

        # Defaults
        shapefunc = (lambda x: x) if shapefunc is None else shapefunc
        dim = 2 if dim is None else dim
        issequence = False if issequence is None else issequence

        # Parse layer spec
        parsey = netutils.parselayerinfo(dim=dim, allowsequences=True, numinp=numinp, issequence=issequence,
                                         inpshape=inpshape)
        self.dim = parsey['dim']
        self.inpdim = parsey['inpdim']
        self.allowsequences = parsey['allowsequences']
        self.issequence = parsey['issequence']

        # Meta
        self.func = func
        self.funcargs = [] if funcargs is None else list(funcargs)
        self.funckwargs = {} if funckwargs is None else dict(funckwargs)

        self.shapefunc = shapefunc
        self.shapefuncargs = [] if shapefuncargs is None else list(shapefuncargs)
        self.shapefunckwargs = {} if shapefunckwargs is None else dict(shapefunckwargs)

        # Structure inference
        self.numinp = parsey['numinp']
        self.numout = numout

        # Shape inference
        self.inpshape = parsey['inpshape']

        # Containers for X and Y
        self.x = pyk.delist([T.tensor('floatX', [False, ] * indim, name='x{}:'.format(inpnum) + str(id(self)))
                             for inpnum, indim in enumerate(pyk.obj2list(self.inpdim))])
        self.y = pyk.delist([T.tensor('floatX', [False, ] * oudim, name='x{}:'.format(outnum) + str(id(self)))
                             for outnum, oudim in enumerate(pyk.obj2list(self.outdim))])


    def feedforward(self, inp=None):
        if inp is None:
            inp = self.x
        else:
            self.x = inp

        # Evaluate function
        y = self.func(inp, *self.funcargs, **self.funckwargs)
        # Convert y to list if it's a tuple
        y = list(y) if isinstance(y, tuple) else y

        # Make sure output is consistent with expectation (since func can do whatever the f it likes)
        len(pyk.obj2list(y)) == self.numout, "The given function must return {} outputs, " \
                                             "got {} instead.".format(self.numout, len(pyk.obj2list(y)))

        self.y = y
        return self.y

    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        outshape = self.shapefunc(inpshape, *self.shapefuncargs, **self.shapefunckwargs)
        return outshape


class lasagnelayer(layer):
    """Class to wrap Theano graphs built with Lasagne."""
    def __init__(self, outputlayers, numinp=1, inpshape=None):
        # Init superclass
        super(lasagnelayer, self).__init__()
        # Make sure lasagne is available
        assert las is not None, "Lasagne could not be imported."

        # Parse layer info
        parsey = netutils.parselayerinfo(dim=2, allowsequences=True, numinp=numinp, issequence=False,
                                         inpshape=inpshape)

        self.dim = parsey['dim']
        self.inpdim = parsey['inpdim']
        self.allowsequences = parsey['allowsequences']
        self.issequence = parsey['issequence']

        pass
    pass