__author__ = 'nasimrahaman'

""" Custom Theano Operations """

# Imports
import theano
import theano.tensor as T
from theano.compile import ViewOp


# Classes
class maskgrads(ViewOp):
    """
    Op to mask gradients of a tensor with a given binary mask. Can be used to kill gradients of a subtensor specified
    by mask.
    """

    # Constructor
    def __init__(self, mask):
        """
        :type mask: theano.tensor.sharedvar.TensorSharedVariable
        :param mask: Binary mask. One -> Gradient not killed, Zero -> Gradient killed.
        """
        self.mask = mask

    def grad(self, args, g_outs):
        return [self.mask * g_out for g_out in g_outs]


class masktens(ViewOp):
    """
    Op to mask a tensor with a given binary mask. Also kills gradients of a subtensor specified
    by mask.
    """

    # Constructor
    def __init__(self, mask):
        """
        :type mask: theano.tensor.sharedvar.TensorSharedVariable
        :param mask: Binary mask. One -> Gradient not killed, Zero -> Gradient killed.
        """
        self.mask = mask

    def perform(self, node, inp, out):
        tensor, = inp
        z, = out
        z[0] = tensor * self.mask

    def grad(self, args, g_outs):
        return [self.mask * g_out for g_out in g_outs]


# Functions
def maskgradient(x, mask):
    """
    :type x: theano.tensor.var.TensorVariable or theano.tensor.sharedvar.TensorSharedVariable
    :param x: Input to be masked
    :type mask: theano.tensor.sharedvar.TensorSharedVariable
    :param mask: Binary mask. One -> Gradient not killed, Zero -> Gradient killed.
    """
    try:
        return maskgrads(mask=mask)(x)
    except ValueError:
        raise ValueError("Gradient Masking Op failed. Sure that the mask and the Op input 'x' can be multiplied?")


# Functions
def masktensor(x, mask):
    """
    :type x: theano.tensor.var.TensorVariable or theano.tensor.sharedvar.TensorSharedVariable
    :param x: Input to be masked
    :type mask: theano.tensor.sharedvar.TensorSharedVariable
    :param mask: Binary mask. One -> Gradient & tensor not killed, Zero -> Gradient & tensor killed.
    """
    try:
        return masktens(mask=mask)(x)
    except ValueError:
        raise ValueError("Tensor Masking Op failed. Sure that the mask and the Op input 'x' can be multiplied?")

# Tests
if __name__ == "__main__" and __debug__:
    import numpy as np

    x = T.matrix('x')
    mask = np.zeros(shape=(2, 2)).astype('float32')
    mask[0, 1] = 1.

    mx = masktensor(x, mask)

    # Compute gradients
    dmx = T.grad(mx.sum(), x)
    # FIXME: Forward pass test failed
    # Backward pass OK.
    # Evaluate
    print(dmx.eval({x: np.ones(shape=(2, 2)).astype('float32')}))

    pass