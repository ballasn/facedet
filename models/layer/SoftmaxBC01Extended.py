import theano
import theano.printing as printing
from pylearn2.models.mlp import *

class SoftmaxExtended(Layer):
    """
    .. todo::

        WRITEME (including parameters list)

    Parameters
    ----------
    n_classes : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_row_norm : WRITEME
    no_affine : WRITEME
    max_col_norm : WRITEME
    init_bias_target_marginals : WRITEME
    """

    def __init__(self, n_classes, layer_name, irange=None,
                 istdev=None,
                 sparse_init=None, W_lr_scale=None,
                 b_lr_scale=None, max_row_norm=None,
                 no_affine=True,
                 max_col_norm=None, init_bias_target_marginals=None,
                 thresholds=[]):

        super(SoftmaxExtended, self).__init__()

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)
        # Recall and precision are linked to their respective thresholds
        thresholds.append(0.5)
        # Keep only unique elements
        thresholds = list(set(thresholds))
        # Sort initially for a better display
        thresholds.sort()

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals

        assert isinstance(n_classes, py_integer_types)



    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        self.output_space = \
        VectorSpace(self.input_space.shape[0]*self.input_space.shape[1]*self.n_classes)

        self.desired_space = self.input_space
        if not isinstance(space, Space):
            raise TypeError("Expected Space, got " +
                            str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, Conv2DSpace)

        desired_dim = self.n_classes

        assert self.input_space.num_channels == 2
        self._params = []



    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        self.desired_space.validate(state_below)
        assert state_below.ndim == 4

        Z = state_below
        e_Z = T.exp(Z - Z.max(axis=1, keepdims=True))
        rval = e_Z / e_Z.sum(axis=1, keepdims=True)

        # Receiving BC01
        #rval_swaped = rval.dimshuffle(0, 2, 3, 1)
        # Use rval shape to override input_shape
        rval_reshaped = rval.reshape(shape=(rval.shape[0],
                                            self.n_classes,
                                            rval.shape[2],
                                            rval.shape[3]),
                                            ndim=4)
        assert rval_reshaped.ndim == 4
        return rval_reshaped

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        #assert hasattr(Y_hat, 'owner')
        #owner = Y_hat.owner
        #assert owner is not None
        #op = owner.op
        #if isinstance(op, Print):
        #    #assert len(owner.inputs) == 1
        #    Y_hat, = owner.inputs
        #    owner = Y_hat.owner
        #    op = owner.op
        #assert isinstance(op, T.nnet.Softmax)
        #assert len(owner.inputs) == 2
        #z,ss = owner.inputs
        z = Y_hat

        # Now zdim ==4 !
        #assert z.ndim == 4


        # Was using input_space.num_channels instead of n_classes
        z = z.reshape(shape=(self.mlp.batch_size*
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1],
                                   self.n_classes ),ndim=2)

        # Was using input_space.num_channels instead of n_classes
        Y = Y.reshape(shape=(self.mlp.batch_size*
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1],
                                   self.n_classes ),ndim=2)


        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1
        rval = log_prob_of.mean()

        return  - rval

    def get_detection_channels_from_state(self, state, target):
        rval = OrderedDict()
        # target is a 128x2 vector
        # p(face), p(non-face) ???
        y_hat = state > 0.5
        y = target > 0.5

        if y.ndim == 2:
            y = y.dimshuffle('x',0,1)
        if y_hat.ndim == 2:
            y_hat = y_hat.dimshuffle('x',0,1)
        real_pos = y.sum(axis=(0,1))[0]
        pred_pos = y_hat.sum(axis=(0,1))[0]
        # Beware that if you mean on all axis
        # You'll get 0.5 as p(face)+p(non-face)=1
        rval['kl'] = self.cost(Y_hat=state, Y=target)

        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)

        # Ground truth positives
        gtp = T.cast(real_pos, 'float32')
        rval['gt_p'] = gtp

        # Results for the asked thresholds
        for e in self.thresholds:
            y_hat_03 = state > e
            tp = (y * y_hat_03).sum(axis=(0,1))
            tp = T.cast(tp[0], 'float32')

            fp = ((1-y) * y_hat_03).sum(axis=(0,1))
            fp = T.cast(fp[0], 'float32')

            precision = tp / T.maximum(1., tp + fp)
            recall = tp / T.maximum(1., gtp)

            rval['tp_'+str(e)] = tp
            rval['fp_'+str(e)] = fp
            rval['precision_'+str(e)] = precision
            rval['recall_'+str(e)] = recall

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        # channels that does not require state information
        rval = OrderedDict()

        if (state_below is not None) or (state is not None):
            # Was using input_space.num_channels instead of n_classes
            state = state.reshape(shape=(self.mlp.batch_size*
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1],
                                   self.input_space.num_channels ),ndim=2)

            mx = state.max(axis=1)

            rval.update(OrderedDict([('mean_max_class', mx.mean()),
                                ('max_max_class', mx.max()),
                                ('min_max_class', mx.min())]))

            assert targets is not None
            # Was using input_space.num_channels instead of n_classes
            targets =\
                targets.reshape(
                    shape=(self.mlp.batch_size*
                           self.input_space.shape[0]*
                           self.input_space.shape[1],
                           self.n_classes), ndim=2)
                           #self.input_space.shape[1],

            #bla_shape = theano.shape(state)

            y_hat = T.argmax(state, axis=1)
            y = T.argmax(targets, axis=1)

            #state = printing.Print('state')(state)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass

            # Was using input_space.num_channels instead of n_classes
            state = state.reshape(shape=(self.mlp.batch_size,
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1]*
                                   self.n_classes ),ndim=2)
            targets = targets.reshape(shape=(self.mlp.batch_size,
                                   self.input_space.shape[0]*
                                   self.input_space.shape[1]*
                                   self.n_classes ),ndim=2)

            rval['nll'] = self.cost(Y_hat=state, Y=targets)
            # Adding the detection info to the monitoring channels
            rval.update(self.get_detection_channels_from_state(state, targets))

        return rval


    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        return 0



    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        return 0

