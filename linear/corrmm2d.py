import functools
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.cuda.blas import GpuCorrMM

from pylearn2.packaged_dependencies.theano_linear.conv2d \
    import Conv2d as OrigConv2D

from pylearn2.linear.linear_transform import LinearTransform as P2LT
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng


default_seed = [2012, 11, 6, 9]
default_sparse_seed = [2012, 11, 6]


class CorrMM2D(OrigConv2D):
    """
    Wrapper on the Theano GpuCorrMM op.
    """

    def __init__(self, filters,
                 batch_size,
                 input_space,
                 output_axes=('b', 'c', 0, 1),
                 subsample=(1, 1),
                 pad=(1,1),
                 border_mode='valid',
                 filters_shape=None,
                 message=''):

        assert batch_size is None or batch_size > 0
        self.input_space = input_space
        self.output_axes = output_axes
        self._pad = pad

        super(Conv2D, self).__init__(
            filters=filters,
            img_shape=(batch_size, input_space.num_channels,
                       input_space.shape[0], input_space.shape[1]),
            subsample=subsample,
            border_mode=border_mode,
            filters_shape=filters.get_value(borrow=True).shape,
            message=message
        )
        if pad == (1, 1):
            self.conv_op = GpuCorrMM(subsample=self._subsample,
                                     pad=border_mode)
        else:
            self.conv_op = GpuCorrMM(subsample=self._subsample, pad=pad)

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return [self._filters]

    @functools.wraps(P2LT.get_weights_topo)
    def get_weights_topo(self, borrow):
        """
        .. todo::

            WRITEME
        """
        return np.transpose(self._filters.get_value(borrow=borrow),
                            (0, 2, 3, 1))

    def lmul(self, x):
        """
        .. todo::

            WRITEME properly

        dot(x, A)

        This method overrides the original Conv2D lmul to make it work
        with arbitrary axis orders
        """

        # x must be formatted as batch index, channel, topo dim 0, topo dim 1
        # for use with conv2d, so check what the current input space format is
        assert x.ndim == 4
        axes = self.input_space.axes
        assert len(axes) == 4

        op_axes = ('b', 'c', 0, 1)

        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                axes.index('b'),
                axes.index('c'),
                axes.index(0),
                axes.index(1))

        rval =  self.conv_op(x, self._filters)

        # Format the output based on the output space
        axes = self.output_axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                op_axes.index(axes[0]),
                op_axes.index(axes[1]),
                op_axes.index(axes[2]),
                op_axes.index(axes[3])
            )

        return rval

    def set_batch_size(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        self._img_shape = tuple([batch_size] + list(self._img_shape[1:]))


def make_random_conv2D(irange, input_space, output_space,
                       kernel_shape, batch_size=None, \
                       subsample = (1,1),
                       pad = (1,1),
                       border_mode = 'valid',
                       message = "", rng = None):
    """
    .. todo::

        WRITEME properly

    Creates a Conv2D with random kernels
    """

    rng = make_np_rng(rng, default_seed, which_method='uniform')

    W = sharedX(rng.uniform(
        -irange, irange,
        (output_space.num_channels, input_space.num_channels,
         kernel_shape[0], kernel_shape[1])
    ))

    return Conv2D(
        filters=W,
        batch_size=batch_size,
        input_space=input_space,
        output_axes=output_space.axes,
        subsample=subsample, border_mode=border_mode, pad=pad,
        filters_shape=W.get_value(borrow=True).shape, message=message
    )


