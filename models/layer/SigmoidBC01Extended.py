import theano
import theano.printing as printing
from pylearn2.models.mlp import *

import theano.tensor as T
from pylearn2.expr.nnet import (elemwise_kl, kl, compute_precision,
                                    compute_recall, compute_f1)


class SigmoidExtended(Layer):
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

    def __init__(self, n_classes,
                 layer_name,
                 thresholds=[]):

        super(SigmoidExtended, self).__init__()

        no_affine = True,

        # Recall and precision are linked to their respective thresholds
        thresholds.append(0.5)
        # Keep only unique elements
        thresholds = list(set(thresholds))
        # Sort initially for a better display
        thresholds.sort()

        monitor_style='classification'

        self.__dict__.update(locals())
        del self.self

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
        p = T.nnet.sigmoid(state_below.dimshuffle(2, 3, 0, 1))
        return p

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        """
        Returns a batch (vector) of
        mean across units of KL divergence for each example.

        Parameters
        ----------
        Y : theano.gof.Variable
            Targets
        Y_hat : theano.gof.Variable
            Output of `fprop`

        mean across units, mean across batch of KL divergence
        Notes
        -----
        Uses KL(P || Q) where P is defined by Y and Q is defined by Y_hat
        Currently Y must be purely binary. If it's not, you'll still
        get the right gradient, but the value in the monitoring channel
        will be wrong.
        Y_hat must be generated by fprop, i.e., it must be a symbolic
        sigmoid.

        p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        For binary p, some terms drop out:
        - p log q - (1-p) log (1-q)
        - p log sigmoid(z) - (1-p) log sigmoid(-z)
        p softplus(-z) + (1-p) softplus(z)
        """
        total = self.kl(Y=Y, Y_hat=Y_hat)
        ave = total.mean()
        return ave

    def kl(self, Y, Y_hat):
        """
        Computes the KL divergence.


        Parameters
        ----------
        Y : Variable
            targets for the sigmoid outputs. Currently Y must be purely binary.
            If it's not, you'll still get the right gradient, but the
            value in the monitoring channel will be wrong.
        Y_hat : Variable
            predictions made by the sigmoid layer. Y_hat must be generated by
            fprop, i.e., it must be a symbolic sigmoid.

        Returns
        -------
        ave : Variable
            average kl divergence between Y and Y_hat.

        Notes
        -----
        Warning: This function expects a sigmoid nonlinearity in the
        output layer and it uses kl function under pylearn2/expr/nnet/.
        Returns a batch (vector) of mean across units of KL
        divergence for each example,
        KL(P || Q) where P is defined by Y and Q is defined by Y_hat:

        p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        For binary p, some terms drop out:
        - p log q - (1-p) log (1-q)
        - p log sigmoid(z) - (1-p) log sigmoid(-z)
        p softplus(-z) + (1-p) softplus(z)
        """
        batch_axis = self.output_space.get_batch_axis()
        div = kl(Y=Y, Y_hat=Y_hat, batch_axis=batch_axis)
        return div

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        rval = elemwise_kl(Y, Y_hat)
        assert rval.ndim == 2
        return rval

    def get_detection_channels_from_state(self, state, target):
        """
        Returns monitoring channels when using the layer to do detection
        of binary events.

        Parameters
        ----------
        state : theano.gof.Variable
            Output of `fprop`
        target : theano.gof.Variable
            The targets from the dataset

        Returns
        -------
        channels : OrderedDict
            Dictionary mapping channel names to Theano channel values.
        """

        rval = OrderedDict()
        y_hat = state > 0.5
        y = target > 0.5
        wrong_bit = T.cast(T.neq(y, y_hat), state.dtype)
        rval['01_loss'] = wrong_bit.mean()
        rval['kl'] = self.cost(Y_hat=state, Y=target)

        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)
        tp = (y * y_hat).sum()
        fp = ((1-y) * y_hat).sum()

        precision = compute_precision(tp, fp)
        recall = compute_recall(y, tp)
        f1 = compute_f1(precision, recall)

        rval['precision'] = precision
        rval['recall'] = recall
        rval['f1'] = f1

        tp = (y * y_hat).sum(axis=0)
        fp = ((1-y) * y_hat).sum(axis=0)

        precision = compute_precision(tp, fp)

        rval['per_output_precision_max'] = precision.max()
        rval['per_output_precision_mean'] = precision.mean()
        rval['per_output_precision_min'] = precision.min()

        recall = compute_recall(y, tp)

        rval['per_output_recall_max'] = recall.max()
        rval['per_output_recall_mean'] = recall.mean()
        rval['per_output_recall_min'] = recall.min()

        f1 = compute_f1(precision, recall)

        rval['per_output_f1_max'] = f1.max()
        rval['per_output_f1_mean'] = f1.mean()
        rval['per_output_f1_min'] = f1.min()

        return rval

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        rval = OrderedDict()
        if target is not None:
            if self.monitor_style == 'detection':
                rval.update(self.get_detection_channels_from_state(state,
                                                                   target))
            else:
                assert self.monitor_style == 'classification'
                # Threshold Y_hat at 0.5.
                prediction = T.gt(state, 0.5)
                # If even one feature is wrong for a given training example,
                # it's considered incorrect, so we max over columns.
                incorrect = T.neq(target, prediction).max(axis=1)
                rval['misclass'] = T.cast(incorrect, config.floatX).mean()

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):


        rval = OrderedDict()
        if (targets is not None) and \
                ((state_below is not None) or (state is not None)):
            if state is None:
                state = self.fprop(state_below)
            if self.monitor_style == 'detection':
                rval.update(self.get_detection_channels_from_state(state,
                                                                   targets))
            else:
                assert self.monitor_style == 'classification'
                # Threshold Y_hat at 0.5.
                prediction = T.gt(state, 0.5)
                # If even one feature is wrong for a given training example,
                # it's considered incorrect, so we max over columns.
                incorrect = T.neq(targets, prediction).max(axis=1)
                rval['misclass'] = T.cast(incorrect, config.floatX).mean()
        return rval