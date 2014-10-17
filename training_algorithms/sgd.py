import logging
import warnings
import numpy as np

from theano import config
from theano import function
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values

from pylearn2.monitor import Monitor
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor \
        as LRMomentumAdjustor
from pylearn2.utils.iteration import is_stochastic, has_uniform_batch_size
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.utils import contains_nan
from pylearn2.utils import contains_inf
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.timing import log_timing
from pylearn2.utils.rng import make_np_rng



class MonitorBasedLRDecay(TrainExtension):
    """
    A TrainExtension that uses the on_monitor callback to adjust
    the learning rate. It pulls out a channel
    from the model's monitor and adjusts the learning rate
    based on what happened to the monitoring channel on a
    certain number of epochs.
    If the channel did not increase during the
    nb_epoch, the learning rate is scaled by shrink_lr

    channel_name : str, optional
        If specified, use channel_name as the channel to guide the
        learning rate adaptation. Conflicts with dataset_name.
        If neither dataset_name nor channel_name is specified, uses
        "objective"
    """

    def __init__(self,
                 nb_epoch,
                 shrink_lr,
                 min_lr,
                 channel_name):
        self.nb_epoch = nb_epoch
        self.shrink_lr = shrink_lr
        self.min_lr = min_lr
        self.channel_name = channel_name
        assert self.channel_name is not None
        self._count = 0
        self._min_v = 0

    def on_monitor(self, model, dataset, algorithm):
        """
        Adjusts the learning rate based on the contents of model.monitor

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        algorithm : WRITEME
        """
        model = algorithm.model
        lr = algorithm.learning_rate
        current_learning_rate = lr.get_value()
        assert hasattr(model, 'monitor'), ("no monitor associated with "
                                           + str(model))
        monitor = model.monitor
        monitor_channel_specified = True

        try:
            v = monitor.channels[self.channel_name].val_record
        except KeyError:
            err_input = ''
            err_input = 'The channel_name \'' + str(
                self.channel_name) + '\' is not valid.'
            err_message = 'There is no monitoring channel named \'' + \
                str(self.channel_name) + '\'. You probably need to ' + \
                'specify a valid monitoring channel by using either ' + \
                'dataset_name or channel_name in the ' + \
                'MonitorBasedLRDecay constructor. ' + err_input
            reraise_as(ValueError(err_message))

        if len(v) == 1:
            #only the initial monitoring has happened
            #no learning has happened, so we can't adjust the learning rate yet
            #just do nothing
            self._min_v = v[0]
            return

        rval = current_learning_rate
        log.info("monitoring channel is {0}".format(self.channel_name))

        if v[-1] < self._min_v:
            self._min_v = v[-1]
            self._count = 0
        else:
            self._count += 1

        if self._count > self.nb_epoch:
            self._count = 0
            rval = self.shrink_lr * rval

        rval = max(self.min_lr, rval)
        lr.set_value(np.cast[lr.dtype](rval))

