__author__ = 'nasimrahaman'

""" Custom Theano Operations """

# Imports
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