
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
from pylearn2.models.mlp import Layer

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


class LayerList(Layer):
    """
    FIXME
    """

    def __init__(self, layer_name, layer_list):
        super(LayerList).__init__()
        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        rval = state_below
        for l in self.layerlist:
            rval = l.fprop(rval)
        return rval


    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return 0

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):
        return 0

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        ### set input space for each sublayer
        self.input_space = space
        output = self.input_space
        for l in self.layerlist:
            l.set_input_space(output)
            output = l.output_space
        self.output_space = output

    @wraps(Layer.get_params)
    def get_params(self):
        rval = []
        for layer in self.layers:
            rval = safe_union(layer.get_params(), rval)
        return rval


    def _weight_decay_aggregate(self, method_name, coeff):
        if isinstance(coeff, py_float_types):
            return T.sum([getattr(layer, method_name)(coeff)
                          for layer in self.layerlist])
        elif is_iterable(coeff):
            assert all(layer_coeff >= 0 for layer_coeff in coeff)
            return T.sum([getattr(layer, method_name)(layer_coeff) for
                          layer, layer_coeff in safe_zip(self.layerlist, coeff)
                          if layer_coeff > 0], dtype=config.floatX)
        else:
            raise TypeError("LayerList's " + method_name + " received "
                            "coefficients of type " + str(type(coeff)) + " "
                            "but must be provided with a float or list/tuple")

    def get_weight_decay(self, coeff):
        return self._weight_decay_aggregate('get_weight_decay', coeff)

    def get_l1_weight_decay(self, coeff):
        return self._weight_decay_aggregate('get_l1_weight_decay', coeff)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        return sum(layer.cost(Y_elem, Y_hat_elem)
                   for layer, Y_elem, Y_hat_elem in
                   safe_zip(self.layerlist, Y, Y_hat))

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):
        super(LayerList, self).set_mlp(mlp)
        for layer in self.layerlist:
            layer.set_mlp(mlp)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        rval = OrderedDict()
        # TODO: reduce redundancy with fprop method
        cur_state_below = state_below
        for i, layer in enumerate(self.layerlist):
            if state is not None:
                cur_state = state[i]
            else:
                cur_state = None
            if targets is not None:
                cur_targets = targets[i]
            else:
                cur_targets = None
            d = layer.get_layer_monitoring_channels(cur_state_below,
                                                    cur_state, cur_targets)
            for key in d:
                rval[layer.layer_name + '_' + str(i) + '_' +  key] = d[key]
            cur_state_below = layer.fprop(state_below)

        return rval

    @wraps(Model._modify_updates)
    def _modify_updates(self, updates):
        for layer in self.layerlist:
            layer.modify_updates(updates)

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        return get_lr_scalers_from_layers(self)
