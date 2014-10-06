"""
PoolUnit, modified to support variable sizes as input
"""

import logging
import math
import sys
import warnings

import numpy as np
from theano import config, scan
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
import theano.tensor as T
from theano.tensor.signal import downsample

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
from pylearn2.models.mlp import Layer, max_pool, mean_pool
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



class PoolUnit(Layer):
    """
    FIXME
    """
    def __init__(self,
                 layer_name,
                 pool_type=None,
                 pool_shape=None,
                 pool_stride=None,
                 output_normalization=None,
                 monitor_style="classification"):

        if pool_type is not None:
            assert pool_shape is not None, ("You should specify the shape of "
                                           "the spatial %s-pooling." % pool_type)
            assert pool_stride is not None, ("You should specify the strides of "
                                            "the spatial %s-pooling." % pool_type)

        super(PoolUnit, self).__init__()
        self.__dict__.update(locals())
        assert monitor_style in ['classification',
                            'detection'], ("%s.monitor_style"
                            "should be either detection or classification"
                            % self.__class__.__name__)
        del self.self

    def initialize_output_space(self):
        """
        Initializes the output space of the ConvElemwise layer by taking
        pooling operator and the hyperparameters of the convolutional layer
        into consideration as well.
        """
        dummy_batch_size = self.mlp.batch_size
        if dummy_batch_size is None:
            dummy_batch_size = 2

        self.output_channels = self.input_space.num_channels

        dummy_detector = \
            sharedX(self.input_space.get_origin_batch(dummy_batch_size))
        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(bc01=dummy_detector,
                                   pool_shape=self.pool_shape,
                                   pool_stride=self.pool_stride,
                                   image_shape=self.input_space.shape)
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(bc01=dummy_detector,
                                    pool_shape=self.pool_shape,
                                    pool_stride=self.pool_stride,
                                    image_shape=self.input_space.shape)


            dummy_p = dummy_p.eval()
            print dummy_detector.eval().shape, dummy_p.shape
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                                   dummy_p.shape[3]],
                                            num_channels=
                                                self.output_channels,
                                            axes=('b', 'c', 0, 1))
        else:
            dummy_detector = dummy_detector.eval()
            print dummy_detector.shape
            self.output_space = Conv2DSpace(shape=[dummy_detector.shape[2],
                                            dummy_detector.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
        print "Output shape", self.output_space.shape, self.output_space.num_channels

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.get_mlp().rng
        self.initialize_output_space()


    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        None

    @wraps(Layer.get_params)
    def get_params(self):
        return []

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        return []

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        return []



    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):

        rval = super(ConvElemwise, self).get_monitoring_channels_from_state(state,
                                                                            target)

        cst = self.cost
        orval = self.nonlin.get_monitoring_channels_from_state(state,
                                                               target,
                                                               cost_fn=cst)

        rval.update(orval)

        return rval

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        return OrderedDict()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        return OrderedDict()


    @wraps(Layer.fprop)
    def fprop(self, state_below):

        d = state_below
        if self.pool_type is not None:
            # Format the input to be supported by max pooling
            assert self.pool_type in ['max', 'mean'], ("pool_type should be"
                                                      "either max or mean"
                                                      "pooling.")

            if self.pool_type == 'max':
                p = downsample.max_pool_2d(d,
                                           self.pool_shape,
                                           ignore_border=False)

                #p = max_pool(bc01=d, pool_shape=self.pool_shape,
                #        pool_stride=self.pool_stride,
                #        image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)

            self.output_space.validate(p)
        else:
            p = d

        if not hasattr(self, 'output_normalization'):
           self.output_normalization = None

        if self.output_normalization:
           p = self.output_normalization(p)
        return p

    def cost(self, Y, Y_hat):
        """
        Cost for convnets is hardcoded to be the cost for sigmoids.
        TODO: move the cost into the non-linearity class.

        Parameters
        ----------
        Y : theano.gof.Variable
            Output of `fprop`
        Y_hat : theano.gof.Variable
            Targets

        Returns
        -------
        cost : theano.gof.Variable
            0-D tensor describing the cost

        Notes
        -----
        Cost mean across units, mean across batch of KL divergence
        KL(P || Q) where P is defined by Y and Q is defined by Y_hat
        KL(P || Q) = p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        """
        assert self.nonlin.non_lin_name == "sigmoid", ("ConvElemwise "
                                                       "supports "
                                                       "cost function "
                                                       "for only "
                                                       "sigmoid layer "
                                                       "for now.")
        batch_axis = self.output_space.get_batch_axis()
        ave_total = kl(Y=Y, Y_hat=Y_hat, batch_axis=batch_axis)
        ave = ave_total.mean()
        return ave


