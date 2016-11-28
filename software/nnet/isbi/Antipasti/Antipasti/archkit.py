__author__ = "Nasim Rahaman"

__doc__ = """ This module contains architectural layers for building networks with general topologies."""

# Imports
import theano as th
import theano.tensor as T
import numpy as np

import copy

from netkit import layer
import pykit as pyk


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

        return outshape

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

    def inferoutshape(self, inpshape=None, checkinput=False):
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
        :type activation: callable
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
        self.activation = activation
        self.dim = dim if dim is not None else {4: 2, 5: 3}[len(inpshape)]
        self.allowsequences = True
        self.issequence = self.dim == 2 and len(inpshape) == 5 if issequence is None else issequence
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

        self.y = self.activation(inp)

        return self.y

    def inferoutshape(self, inpshape=None, checkinput=False):
        if inpshape is None:
            inpshape = self.inpshape

        return inpshape
