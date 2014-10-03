import logging
import math
import sys
import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.monitor import get_monitor_doc
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace, IndexSpace
from pylearn2.utils import function
from pylearn2.utils import is_iterable
from pylearn2.utils import py_float_types
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils import contains_nan
from pylearn2.utils import contains_inf
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping

from pylearn2.expr.nnet import (elemwise_kl, kl, compute_precision,
                                    compute_recall, compute_f1)

# Only to be used by the deprecation warning wrapper functions
from pylearn2.costs.mlp import L1WeightDecay as _L1WD
from pylearn2.costs.mlp import WeightDecay as _WD
from pylearn2.models.mlp import ConvNonlinearity

logger = logging.getLogger(__name__)

logger.debug("MLP changing the recursion limit.")
# We need this to be high enough that the big theano graphs we make
# when doing max pooling via subtensors don't cause python to complain.
# python intentionally declares stack overflow well before the stack
# segment is actually exceeded. But we can't make this value too big
# either, or we'll get seg faults when the python interpreter really
# does go over the stack segment.
# IG encountered seg faults on eos3 (a machine at LISA labo) when using
# 50000 so for now it is set to 40000.
# I think the actual safe recursion limit can't be predicted in advance
# because you don't know how big of a stack frame each function will
# make, so there is not really a "correct" way to do this. Really the
# python interpreter should provide an option to raise the error
# precisely when you're going to exceed the stack segment.
sys.setrecursionlimit(40000)


class MaxoutBC01(ConvNonlinearity):
    """
    A simple rectifier nonlinearity class for convolutional layers.

    Parameters
    ----------
    num_pieces : int
        The number of different parts in which the output is divided.
    """
    def __init__(self, num_pieces):
        """
        Parameters
        ----------
        num_pieces : int
            The number of different parts in which the output is divided.
        """
        self.non_lin_name = "maxout"
        self.num_pieces = num_pieces

    @wraps(ConvNonlinearity.apply)
    def apply(self, linear_response):
        """
        Applies the rectifier nonlinearity over the convolutional layer.
        """
        s = None
        for i in xrange(self.num_pieces):
            t = linear_response[:, i::self.num_pieces, :, :]
            if s is None:
                s = t
            else:
                s = T.maximum(s, t)

        return s
