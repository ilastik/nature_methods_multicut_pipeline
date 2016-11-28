__doc__ = """"Unified Theano/Tensorflow backend."""

import sys
from warnings import warn

import numpy as np

import theano as th
import theano.tensor as T
from theano.tensor.nnet import conv3d2d


# Flag to indicate which backend is being used
BACKENDNAME = 'THEANO'

if BACKENDNAME == 'THEANO':
    _backend = th
    _backendtensor = T


class backend(object):

    def __theano__conv(self, inp, filters, stride=None, padding=None, bias=None, filtergradmask=None,
                       biasgradmask=None, filtermask=None, biasmask=None, filtergradclips=None, biasgradclips=None,
                       dim=None, convmode='same', issequence=False, implementation='auto'):

        # Do imports locally to prevent circular dependencies
        import netutils as nu
        import theanops as tho
        import pykit as pyk

        # Determine the dimensionality of convolution (2 or 3?)
        if dim is None:
            dim = 3 if not issequence and len(filters.get_value().shape) == 5 and inp.ndim == 5 else 2

        # Smart fix: if convmode is 'same', stride != 1 and padding is None: set automagically set padding.

        # Defaults
        padding = [[0, 0]] * dim if padding is None else padding
        stride = [1] * dim if stride is None else stride
        filtergradclips = [-np.inf, np.inf] if filtergradclips is None else list(filtergradclips)
        biasgradclips = [-np.inf, np.inf] if biasgradclips is None else list(biasgradclips)

        # Autofix inputs
        if isinstance(padding, int):
            padding = [padding] * dim
        if not pyk.islistoflists(pyk.obj2list(padding)):
            padding = [[padval] * dim for padval in pyk.obj2list(padding)]
        if isinstance(stride, int):
            stride = [stride] * dim

        # TODO: Tests
        pass

        # Reshape 2D sequential data if required
        # Log input shape
        inpshape = inp.shape
        reallyissequential = issequence and inp.ndim == 5
        if issequence:
            if reallyissequential:
                inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)
                stride = stride[0:2]
                padding = padding[0:2]
                # TODO: Get rid of these restrictions
                assert stride == [1, 1], "Strided convolution is not implemented for sequential data."
                assert convmode == 'same', "Convmode must be 'same' for sequential data."

            else:
                warn("Expected 5D sequential output, but got 4D non-sequential instead.")

        # Apply gradient masks if required
        if filtergradmask is not None:
            filters = tho.maskgradient(filters, filtergradmask)
        if biasgradmask is not None and bias is not None:
            bias = tho.maskgradient(bias, biasgradmask)

        # Apply masks if required
        if filtermask is not None:
            filters = filtermask * filters
        if biasmask is not None:
            bias = biasmask * bias

        # Determine border_mode for CuDNN/3D conv
        autopaddable, bordermode, trim = self.__theano__bordermode(convmode, padding, filters.get_value().shape)

        # Pad input if required (warn that it's ridiculously slow)
        if not autopaddable and not all([padval == 0 for padval in pyk.flatten(padding)]):
            if not th.config.device == 'cpu' and not self.cpupadwarned:
                warn("Padding might occur on the CPU, which tends to slow things down.")
                self.cpupadwarned = True
            if not isinstance(bordermode, str) and pyk.islistoflists(bordermode):
                # Override padding for 3D convolutions
                inp = nu.pad(inp, bordermode)
                bordermode = 'valid'
            else:
                inp = nu.pad(inp, padding)


        # Convolve 2D (with gradmask + bias), reshape sequential data
        if dim == 2:
            if implementation == 'auto':
                # Convolve
                y = T.nnet.conv2d(input=inp, filters=th.gradient.grad_clip(filters, *filtergradclips),
                                  border_mode=tuple(bordermode) if isinstance(bordermode, list) else bordermode,
                                  filter_shape=filters.get_value().shape, subsample=tuple(stride))
            else:
                raise NotImplementedError("Implementation {} is not implemented.".format(implementation))

            # Trim if required
            if trim:
                y = self.__theano__convtrim(inp=y, filtershape=filters.get_value().shape)

            # Add bias if required
            if bias is not None:
                y = y + th.gradient.grad_clip(bias, *biasgradclips).dimshuffle('x', 0, 'x', 'x')

        elif dim == 3:
            # Convolve 3D (with bias)
            if implementation == 'auto' or implementation == 'conv2d':

                assert stride == [1, 1, 1], "Implementation 'conv2d' does not support strided convolution in 3D."
                assert convmode == 'valid', "Implementation 'conv2d' only supports 'valid' convolutions."

                y = T.nnet.conv3d2d.conv3d(signals=inp, filters=th.gradient.grad_clip(filters, *filtergradclips),
                                           border_mode=bordermode,
                                           filters_shape=filters.get_value().shape)
            else:
                raise NotImplementedError("Implementation {} is not implemented.".format(implementation))

            # Trim if required
            if trim:
                y = self.__theano__convtrim(inp=y, filtershape=filters.get_value().shape)

            # Add bias if required
            if bias is not None:
                y = y + th.gradient.grad_clip(bias, *biasgradclips).dimshuffle('x', 'x', 0, 'x', 'x')

        else:
            raise NotImplementedError("Convolution is implemented in 2D and 3D.")

        # Reshape sequential data
        if issequence and reallyissequential:
            y = y.reshape((inpshape[0], inpshape[1], filters.get_value().shape[0], inpshape[3], inpshape[4]), ndim=5)

        # Return
        return y

    def __theano__pool(self, inp, ds, stride=None, padding=None, poolmode='max', dim=None,
                       ignoreborder=True, issequence=False):

        # Do imports locally to prevent circular dependencies
        import netutils as nu
        import pykit as pyk
        from theano.tensor.signal import downsample

        # Determine the dimensionality of convolution (2 or 3?)
        if dim is None:
            dim = 3 if not issequence and len(ds) == 3 and inp.ndim == 5 else 2

        # Defaults
        poolmode = 'average_exc_pad' if poolmode in ['mean', 'average', 'average_exc_pad'] else poolmode
        padding = [[0, 0]] * dim if padding is None else padding
        stride = ds if stride is None else stride

        # Autofix inputs
        if isinstance(padding, int):
            padding = [padding] * dim
        if not pyk.islistoflists(pyk.obj2list(padding)):
            padding = [[padval] * dim for padval in pyk.obj2list(padding)]
        if isinstance(stride, int):
            stride = [stride] * dim

        # Check if theano can pad input as required
        autopaddable = all([all([dimpad == pad[0] for dimpad in pad]) for pad in padding])

        # Reshape 2D sequential data if required
        # Log input shape
        inpshape = inp.shape
        reallyissequential = issequence and inp.ndim == 5
        if issequence:
            if reallyissequential:
                # Sequential input must be paddable by theano. This is required to reshape the sequential input back to
                # its original shape after pooling.
                assert autopaddable, "Sequential inputs must be paddable by theano. Provided padding {} cannot be " \
                                     "handled at present.".format(padding)
                inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)
                ds = ds[0:2]
                stride = stride[0:2]
                padding = padding[0:2]
            else:
                warn("Expected 5D sequential output, but got 4D non-sequential instead.")

        # Determine what theano needs to be told about how to pad the input
        if autopaddable:
            autopadding = tuple([pad[0] for pad in padding])
        else:
            autopadding = (0,) * dim

        if not autopaddable and not all([padval == 0 for padval in pyk.flatten(padding)]):
            if not th.config.device == 'cpu' and not self.cpupadwarned:
                warn("Padding might occur on the CPU, which tends to slow things down.")
                self.cpupadwarned = True
            inp = nu.pad(inp, padding)

        if dim == 2:
            y = downsample.max_pool_2d(input=inp, ds=ds, st=stride, padding=autopadding, ignore_border=ignoreborder,
                                       mode=poolmode)

        elif dim == 3:
            # parse downsampling ratio, stride and padding
            dsyx = ds[0:2]
            styx = stride[0:2]
            padyx = autopadding[0:2]

            ds0z = (1, ds[2])
            st0z = (1, stride[2])
            pad0z = (0, autopadding[2])

            # Dowsnample yx
            H = downsample.max_pool_2d(input=inp, ds=dsyx, st=styx, padding=padyx, mode=poolmode)
            # Rotate tensor
            H = H.dimshuffle(0, 2, 3, 4, 1)
            # Downsample 0z
            H = downsample.max_pool_2d(input=H, ds=ds0z, st=st0z, padding=pad0z, mode=poolmode)
            # Undo rotate tensor
            y = H.dimshuffle(0, 4, 1, 2, 3)

        else:
            raise NotImplementedError("Pooling is implemented in 2D and 3D.")

        if issequence and reallyissequential:
            # Compute symbolic pool output length
            if ignoreborder:
                pooleny, poolenx = \
                    [T.floor((inpshape[tensorindex] + 2 * autopadding[index] - ds[index] + stride[index])/stride[index])
                     for index, tensorindex in enumerate([3, 4])]
            else:
                poolen = [None, None]

                for index, tensorindex in enumerate([3, 4]):
                    if stride[index] >= ds[index]:
                        poolen[index] = T.floor((inpshape[tensorindex] + stride[index] - 1)/stride[index])
                    else:
                        plen = T.floor((inpshape[tensorindex] - ds[index] + stride[index] - 1)/stride[index])
                        poolen[index] = T.switch(plen > 0, plen, 0)

                pooleny, poolenx = poolen

            y = y.reshape((inpshape[0], inpshape[1], inpshape[2], pooleny, poolenx), ndim=5)

        return y

    def __theano__unpool(self, inp, us, dim=None, issequence=False):

        # Determine the dimensionality of convolution (2 or 3?)
        if dim is None:
            dim = 3 if not issequence and len(us) == 3 and inp.ndim == 5 else 2

        # Reshape 2D sequential data if required
        # Log input shape
        inpshape = inp.shape
        reallyissequential = issequence and inp.ndim == 5
        if issequence:
            if reallyissequential:
                # Reshape
                inp = inp.reshape((inpshape[0] * inpshape[1], inpshape[2], inpshape[3], inpshape[4]), ndim=4)
                us = us[0:2]
            else:
                warn("Expected 5D sequential output, but got 4D non-sequential instead.")

        if dim == 2:
            y = T.repeat(T.repeat(inp, us[0], axis=2), us[1], axis=3)
        elif dim == 3:
            y = T.repeat(T.repeat(T.repeat(inp, us[0], axis=3), us[1], axis=4), us[2], axis=1)
        else:
            raise NotImplementedError("Upsampling is implemented in 2D and 3D.")

        if issequence and reallyissequential:
            # Reshape sequential data (and remember that the spatial size has doubled)
            y = y.reshape((inpshape[0], inpshape[1], inpshape[2], us[0] * inpshape[3], us[1] * inpshape[4]), ndim=5)

        return y

    def __theano__softmax(self, inp, dim=None, predict=False, issequence=False):

        if dim is None:
            assert issequence, "Data dimensionality could not be parsed."
            dim = 2

        # FFD for dimensions 1 and 2
        if dim == 1 or dim == 2:
            # Using the numerically stable implementation (along the channel axis):
            ex = T.exp(inp - T.max(inp, axis=1, keepdims=True))
            y = ex / T.sum(ex, axis=1, keepdims=True)

            # One hot encoding for prediction
            if predict:
                y = T.argmax(y, axis=1)

        elif dim == 3:
            # Stable implementation again, this time along axis = 2 (channel axis)
            ex = T.exp(inp - T.max(inp, axis=2, keepdims=True))
            y = ex / T.sum(ex, axis=2, keepdims=True)

            # One hot encoding for prediction
            if predict:
                y = T.argmax(y, axis=2)

        else:
            raise NotImplementedError("Softmax is implemented in 2D, 3D and 1D.")

        return y

    def __theano__noise(self, inp, noisetype, p=None, n=None, sigma=None, thicken=True, mode=None, srng=None):

        # Local imports
        from theano.tensor.shared_randomstreams import RandomStreams

        # Parse noise type and check arguments
        if noisetype in ['binomial', 'dropout']:
            noisetype = 'binomial'
            assert None not in [n, p], "n and p must be provided for binomial noise."
            mode = 'mul' if mode is None else mode
        elif noisetype in ['gaussian', 'normal']:
            noisetype = 'normal'
            assert sigma is not None, "sigma must be provided for normal noise."
            mode = 'add' if mode is None else mode
        else:
            raise NotImplementedError("Unknown noisetype: {}".format(noisetype))

        # Parse mode
        if mode in ['add', 'additive', 'addition']:
            mode = 'add'
        elif mode in ['mul', 'multiplicative', 'multiplication', 'multiply']:
            mode = 'mul'
        else:
            raise NotImplementedError("Mode {} is not implemented.".format(mode))

        # Default rng
        if srng is None:
            srng = RandomStreams(seed=42)
        elif isinstance(srng, int):
            srng = RandomStreams(seed=srng)

        # Make noise kernel
        if noisetype == 'normal':
            noisekernel = T.cast(srng.normal(size=inp.shape, std=sigma), dtype='floatX')
        elif noisetype == 'binomial':
            noisekernel = T.cast(srng.binomial(size=inp.shape, n=n, p=p), dtype='floatX')
        else:
            raise NotImplementedError

        # Couple with input
        if mode == 'add':
            y = inp + noisekernel
        elif mode == 'mul':
            y = inp * noisekernel
        else:
            raise NotImplementedError

        if thicken and noisetype is 'binomial':
            y = y / getattr(np, th.config.floatX)(p)

        # Return
        return y

    def __theano__batchnorm(self, gamma, beta):
        pass


    @staticmethod
    def __theano__bordermode(convmode, padding, filtershape):
        # Parse dimensionality from filtershape
        dim = {4: 2, 5: 3}[len(filtershape)]
        # 2D case
        if dim == 2:
            # Parse kernel size from filtershape
            kersize = filtershape[2:]

            # Logic to find what bordermode goes in to the conv interface
            # Find out if padding is compatible with DNN
            autopaddable = all([all([dimpad == pad[0] for dimpad in pad]) for pad in padding])
            # Compute the dnn pad value
            if autopaddable:
                dnnpad = [pad[0] for pad in padding]
            else:
                dnnpad = None

            # Whether to trim after conv
            trim = False

            # Get bordermode if padding is [0, 0]
            if dnnpad == [0, 0]:
                if convmode == 'same':
                    if all([ks % 2 == 1 for ks in kersize]):
                        bordermode = 'half'
                    else:
                        bordermode = 'full'
                        trim = True
                elif convmode == 'valid':
                    bordermode = 'valid'
                else:
                    bordermode = 'full'
            elif dnnpad is None:
                if convmode == 'same':
                    bordermode = 'full'
                    trim = True
                elif convmode == 'valid':
                    bordermode = 'valid'
                else:
                    bordermode = 'full'
            else:
                bordermode = dnnpad

            return autopaddable, bordermode, trim
        else:
            # 3D
            import pykit as pyk

            # Check if padding is required. 3D convolution can't do padding, so that needs to be done manually
            autopaddable = all([padval == 0 for padval in pyk.flatten(padding)])

            # Determine bordermode and whether trimming is required
            if convmode == 'same':
                bordermode = 'full'
                trim = True
            elif convmode == 'valid':
                bordermode = 'valid'
                trim = False
            elif convmode == 'full':
                bordermode = 'full'
                trim = False
            else:
                raise NotImplementedError("Border mode {} is not implemented.".format(convmode))

            return autopaddable, bordermode, trim

    @staticmethod
    def __theano__convtrim(inp, filtershape):
        # Get dimensionality from filtershape
        dim = {4: 2, 5: 3}[len(filtershape)]

        if dim == 2:
            trimy, trimx = [(filtershape[i] - 1) / 2 for i in [2, 3]]  # trimmersize = [trimy, trimx]
            offsety, offsetx = [int(1 - (filtershape[i] % 2)) for i in [2, 3]]
            # For nx1 or 1xn convolutions, trimy or trimx is 0. Indexing 0:0 doesn't make sense in python, hence the
            # use of dictionaries
            out = inp[::, ::, (trimy):{-trimy - offsety: -trimy - offsety, 0: None}[-trimy - offsety],
                  (trimx):{-trimx - offsetx: -trimx - offsetx, 0: None}[-trimx - offsetx]]
        # 3D
        elif dim == 3:
            trimz, trimy, trimx = [(filtershape[i] - 1) / 2 for i in [1, 3, 4]]
            offsetz, offsety, offsetx = [int(1 - (filtershape[i] % 2)) for i in [1, 3, 4]]
            out = inp[::, (trimz):{-trimz - offsetz: -trimz - offsetz, 0: None}[-trimz - offsetz], ::,
                  (trimy):{-trimy - offsety: -trimy - offsety, 0: None}[-trimy - offsety],
                  (trimx):{-trimx - offsetx: -trimx - offsetx, 0: None}[-trimx - offsetx]]
        else:
            raise NotImplementedError("Invalid network dimension.")

        return out

    def conv(self, *args, **kwargs):
        if BACKENDNAME == 'THEANO':
            return self.__theano__conv(*args, **kwargs)
        elif BACKENDNAME == 'TENSORFLOW':
            raise NotImplementedError("TF backend is work in progress.")
        else:
            raise NotImplementedError("Unsupported Backend: {}".format(BACKENDNAME))

    def pool(self, *args, **kwargs):
        if BACKENDNAME == 'THEANO':
            return self.__theano__pool(*args, **kwargs)
        elif BACKENDNAME == 'TENSORFLOW':
            raise NotImplementedError("TF backend is work in progress.")
        else:
            raise NotImplementedError("Unsupported Backend: {}".format(BACKENDNAME))

    def unpool(self, *args, **kwargs):
        if BACKENDNAME == 'THEANO':
            return self.__theano__unpool(*args, **kwargs)
        elif BACKENDNAME == 'TENSORFLOW':
            raise NotImplementedError("TF backend is work in progress.")
        else:
            raise NotImplementedError("Unsupported Backend: {}".format(BACKENDNAME))

    def softmax(self, *args, **kwargs):
        if BACKENDNAME == 'THEANO':
            return self.__theano__softmax(*args, **kwargs)
        elif BACKENDNAME == 'TENSORFLOW':
            raise NotImplementedError("TF backend is work in progress.")
        else:
            raise NotImplementedError("Unsupported Backend: {}".format(BACKENDNAME))

    def noise(self, *args, **kwargs):
        if BACKENDNAME == 'THEANO':
            return self.__theano__noise(*args, **kwargs)
        elif BACKENDNAME == 'TENSORFLOW':
            raise NotImplementedError("TF backend is work in progress.")
        else:
            raise NotImplementedError("Unsupported Backend: {}".format(BACKENDNAME))

    def batchnorm(self, *args, **kwargs):
        if BACKENDNAME == 'THEANO':
            return self.__theano__batchnorm(*args, **kwargs)
        elif BACKENDNAME == 'TENSORFLOW':
            raise NotImplementedError("TF backend is work in progress.")
        else:
            raise NotImplementedError("Unsupported Backend: {}".format(BACKENDNAME))

    def __getattr__(self, item):
        # If nothing else works, at least return what the user was looking for...
        # Try to find item in tensor submodule
        if hasattr(_backendtensor, item):
            return getattr(_backendtensor, item)
        elif hasattr(_backend, item):
            # If that fails, try to find item in the main module
            return getattr(_backend, item)
        else:
            raise AttributeError("Backend {} or it's tensor submodule does not have a '{}' attribute.".
                                 format(BACKENDNAME, item))

    # FLAGS
    cpupadwarned = False