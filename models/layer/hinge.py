import functools
import logging
import numpy as np
import warnings

from theano import config
from theano.compat.python2x import OrderedDict
from theano.compat.six.moves import zip as izip
from theano.sandbox import cuda
from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer
from pylearn2.models.model import Model
from pylearn2.space import Space
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX
from pylearn2.utils import wraps


from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
from pylearn2.linear import local_c01b
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
else:
    max_pool_c01b = None
from pylearn2.sandbox.cuda_convnet import check_cuda

from theano.gof import Op, Apply
class Print(Op):
    """ This identity-like Op print as a side effect.

    This identity-like Op has the side effect of printing a message
    followed by its inputs when it runs. Default behaviour is to print
    the __str__ representation. Optionally, one can pass a list of the
    input member functions to execute, or attributes to print.

    @type message: String
    @param message: string to prepend to the output
    @type attrs: list of Strings
    @param attrs: list of input node attributes or member functions to print.
                  Functions are identified through callable(), executed and
                  their return value printed.

    :note: WARNING. This can disable some optimizations!
                    (speed and/or stabilization)

            Detailed explanation:
            As of 2012-06-21 the Print op is not known by any optimization.
            Setting a Print op in the middle of a pattern that is usually
            optimized out will block the optimization. for example, log(1+x)
            optimizes to log1p(x) but log(1+Print(x)) is unaffected by
            optimizations.

    """
    view_map = {0: [0]}

    def __init__(self, message="", attrs=("__str__",)):
        self.message = message
        self.attrs = tuple(attrs)  # attrs should be a hashable iterable

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        print self.message, xin

    def grad(self, input, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __eq__(self, other):
        return (type(self) == type(other) and self.message == other.message
                and self.attrs == other.attrs)

    def __hash__(self):
        return hash(self.message) ^ hash(self.attrs)

    def __setstate__(self, dct):
        dct.setdefault('global_fn', _print_fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)



class L2SquareHinge(Layer):
    """
    A layer that can apply an affine transformation
    and use a l2 regularized square hinge loss.

    Parameters
    ----------
    n_classes : int
        Number of classes for softmax targets.
    layer_name : string
        Name of Softmax layers.
    irange : float
        If specified, initialized each weight randomly in
        U(-irange, irange).
    istdev : float
        If specified, initialize each weight randomly from
        N(0,istdev).
    sparse_init : int
        If specified, initial sparse_init number of weights
        for each unit from N(0,1).
    W_lr_scale : float
        Scale for weight learning rate.
    b_lr_scale : float
        Scale for bias learning rate.
    max_row_norm : float
        Maximum norm for a row of the weight matrix.
    no_affine : boolean
        If True, softmax nonlinearity is applied directly to
        inputs.
    max_col_norm : float
        Maximum norm for a column of the weight matrix.
    init_bias_target_marginals : dataset
        Take the probability distribution of the targets into account to
        intelligently initialize biases.
    binary_target_dim : int, optional
        If your targets are class labels (i.e. a binary vector) then set the
        number of targets here so that an IndexSpace of the proper dimension
        can be used as the target space. This allows the softmax to compute
        the cost much more quickly than if it needs to convert the targets
        into a VectorSpace.
    """

    def __init__(self, n_classes, layer_name,
                 C = 0.1,
                 irange=None, istdev=None,
                 sparse_init=None, W_lr_scale=None,
                 b_lr_scale=None, max_row_norm=None,
                 no_affine=False,
                 max_col_norm=None, init_bias_target_marginals=None,
                 binary_target_dim=None):

        super(L2SquareHinge, self).__init__()

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals

        assert isinstance(n_classes, py_integer_types)

        if binary_target_dim is not None:
            assert isinstance(binary_target_dim, py_integer_types)
            self._has_binary_target = True
            self._target_space = IndexSpace(dim=binary_target_dim,
                                            max_labels=n_classes)
        else:
            self._has_binary_target = False

        self.output_space = VectorSpace(n_classes)

        self.b = sharedX(np.zeros((n_classes,)), name='hinge_b')
        if init_bias_target_marginals:
            y = init_bias_target_marginals.y
            if init_bias_target_marginals.y_labels is None:
                marginals = y.mean(axis=0)
            else:
                # compute class frequencies
                if np.max(y.shape) != np.prod(y.shape):
                    raise AssertionError("Use of "
                                         "`init_bias_target_marginals` "
                                         "requires that each example has "
                                         "a single label.")
            marginals = np.bincount(y.flat)/float(y.shape[0])

            assert marginals.ndim == 1
            b = pseudoinverse_softmax_numpy(marginals).astype(self.b.dtype)
            assert b.ndim == 1
            assert b.dtype == self.b.dtype
            self.b.set_value(b)
        else:
            assert init_bias_target_marginals is None

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        rval = OrderedDict()
        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if self.b_lr_scale is not None:
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        warnings.warn("Layer.get_monitoring_channels is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        W = self.W
        assert W.ndim == 2
        sq_W = T.sqr(W)
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)
        # channels that does not require state information
        W = self.W
        assert W.ndim == 2
        sq_W = T.sqr(W)
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        mx = state.max(axis=1)
        rval.update(OrderedDict([('mean_max_class', mx.mean()),
                            ('max_max_class', mx.max()),
                            ('min_max_class', mx.min())]))
        if target is not None:
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(target, axis=1)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        # channels that does not require state information
        W = self.W
        assert W.ndim == 2
        sq_W = T.sqr(W)
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        if (state_below is not None) or (state is not None):
            if state is None:
                state = self.fprop(state_below)
            mx = state.max(axis=1)
            rval.update(OrderedDict([('mean_max_class', mx.mean()),
                                ('max_max_class', mx.max()),
                               ('min_max_class', mx.min())]))

            if targets is not None:
                y_hat = T.argmax(state, axis=1)
                y = T.argmax(targets, axis=1)
                misclass = T.neq(y, y_hat).mean()
                misclass = T.cast(misclass, config.floatX)
                rval['misclass'] = misclass
                rval['nll'] = self.cost(Y_hat=state, Y=targets)
        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got " +
                            str(space)+" of type "+str(type(space)))
        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)
        desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng
        if self.no_affine:
            self._params = []
        else:
            print (self.input_dim, self.n_classes)
            if self.irange is not None:
                assert self.istdev is None
                assert self.sparse_init is None
                W = rng.uniform(-self.irange,
                                self.irange,
                                (self.input_dim, self.n_classes))
            elif self.istdev is not None:
                assert self.sparse_init is None
                W = rng.randn(self.input_dim, self.n_classes) * self.istdev
            else:
                assert self.sparse_init is not None
                W = np.zeros((self.input_dim, self.n_classes))
                for i in xrange(self.n_classes):
                    for j in xrange(self.sparse_init):
                        idx = rng.randint(0, self.input_dim)
                        while W[idx, i] != 0.:
                            idx = rng.randint(0, self.input_dim)
                        W[idx, i] = rng.randn()

            self.W = sharedX(W,  'hinge_W')

            self._params = [self.b, self.W]

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.np_format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt,
                                         self.input_space.axes,
                                         ('b', 0, 1, 'c'))
        return rval

    @wraps(Layer.get_weights)
    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()
        return self.W.get_value()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):
        self.W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):
        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):
        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):
        return ('v', 'h')

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        ## Precondition
        self.input_space.validate(state_below)
        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)
        self.desired_space.validate(state_below)
        assert state_below.ndim == 2
        assert self.W.ndim == 2

        ## Linear prediction
        rval = T.dot(state_below, self.W) + self.b
        return rval

    def hinge_cost(self, Y, Y_hat):
        ### print size of Y_hat

        #Y = Print(message="Y")(Y)
        #Y_hat = Print(message="Y_hat")(Y_hat)

        prob = (self.C * self.W.norm(2) + (T.maximum(0, 1 - (Y - Y_hat)) ** 2.)).sum(axis=1)
        #.W = Print(message="W")(self.W)
        #prob = (T.maximum(1 - Y * Y_hat, 0) ** 2.).sum(axis=0)
        #prob = Print(message="prob")(prob)
        return prob

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        return self.hinge_cost(Y, Y_hat).mean()

    # @wraps(Layer.cost_matrix)
    # def cost_matrix(self, Y, Y_hat):
    #     # cost = self._cost(Y, Y_hat)
    #     # if self._has_binary_target:
    #     #     flat_Y = Y.flatten()
    #     #     flat_matrix = T.alloc(0, (Y.shape[0]*cost.shape[1]))
    #     #     flat_indices = flat_Y + T.extra_ops.repeat(
    #     #         T.arange(Y.shape[0])*cost.shape[1], Y.shape[1]
    #     #     )
    #     #     cost = T.set_subtensor(flat_matrix[flat_indices], flat_Y)

    #     # return cost
    #     return None

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * abs(W).sum()

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        if self.no_affine:
            return
        if self.max_row_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = updated_W * scales.dimshuffle(0, 'x')
        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))
