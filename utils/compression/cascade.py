import logging
import math
import operator
import sys
import warnings


from pylearn2.compat import OrderedDict
from pylearn2.utils import wraps
from pylearn2.models.mlp import Layer
from pylearn2.costs.mlp import Default
from pylearn2.space import CompositeSpace
from pylearn2.models.model import Model

import theano.tensor as T
from theano import config
from theano.ifelse import ifelse


# Add cascade cost

class Cascade(Model):

    """
    A class representing a cascade of MLPs.

    Parameters
    ----------
    models : list
        A list of mlps.
    thresholds:  list.
	A list of thresholds to decide whether to keep propagating through the cascade. Default value is None.
    batch_size: int
	A integer determining the batch size to be used by all models in the cascade at training. Default value is 128.
    noise: WRITEME, optional
        A list of Space objects specifying the kind of input each model accepts. Default is None.
     monitor_targets : bool, optional
        Default: True
        If true, includes monitoring channels that are functions of the
        targets. This can be disabled to allow monitoring on monitoring
        datasets that do not include targets
    kwargs : dict
        Passed on to the superclass.
    """

    def __init__(self, models,
		 batch_size = 128,
		 thresholds = None,
		 noise = None,
		 monitor_targets = True,
                 **kwargs):
        super(Cascade, self).__init__(**kwargs)

        if thresholds is not None:
	  assert(len(models) == len(thresholds))
	else:
	  assert(len(models) - 1 == 0)

	self.models = models

	self.thresholds = thresholds

	self.batch_size = batch_size
	self.force_batch_size = batch_size

	if noise is not None:
	  self.noise = noise

	self.monitor_targets = True


    def set_batch_size(self, batch_size):

        self.batch_size = batch_size
        self.force_batch_size = batch_size

	for model in self.models:
	  model.set_batch_size(batch_size)


    def fprop(self, state_below, return_all=False):

        ### FIXME precondition on state_below
        if not hasattr(self.models[0], "input_space"):
            raise AttributeError("Input space has not been provided for model 0.")

        output_list = []
        thresh_list = []
        for i in xrange(len(self.models)):
	  output_list.append(self.models[i].fprop(state_below[i]))
	  thresh_list.append(output_list[i] > self.thresholds[i])

        rval = output_list[-1]
        for i in xrange(len(output_list) - 1, 0, -1):
            #print i
            #rval  = thresh_list[i-1] * rval  + (1-thresh_list[i-1]) * output_list[i-1]
            rval  = (output_list[i-1]) * rval  + (1-output_list[i-1]) * output_list[i-1]
            #rval += output_list[i-1]

        return rval


    @wraps(Layer.get_output_space)
    def get_output_space(self):
	# all models must have the same output space
	for i in range(len(self.models[1:])):
	  assert(self.models[i-1].layers[-1].get_output_space() == self.models[i].layers[-1].get_output_space())
        return self.models[-1].layers[-1].get_output_space()


    @wraps(Layer.get_target_space)
    def get_target_space(self):
	# all models must have the same target space
	for i in range(len(self.models[1:])):
	  assert(self.models[i-1].layers[-1].get_target_space() == self.models[i].layers[-1].get_target_space())

        return self.models[-1].layers[-1].get_target_space()

    def get_input_space(self):
	model_input_space = list()
        for model in self.models:
	  model_input_space.append(model.get_input_space())
        return CompositeSpace(model_input_space)

    @wraps(Layer.set_input_space)
    def set_input_space(self, spaces):
	# each model has its own input space
        for i in range(len(self.models)):
	  self.models[i].set_input_space(spaces[i])

    def _update_layer_input_spaces(self):
        """
        Tells each layer of each model what its input space should be.

        Notes
        -----
        This usually resets the layer's parameters!
        """
        for model in self.models:
            model._update_layer_input_spaces()


    def freeze(self, parameter_set):
        """
        Freezes some of the parameters (new theano functions that implement
        learning will not use them; existing theano functions will continue
        to modify them).

        Parameters
        ----------
        parameter_set : set
            Set of parameters to freeze.
        """
        for model in self.models:
            model.freeze_set = model.freeze_set.union(parameter_set)


    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):

	rval = OrderedDict()
        for i in xrange(len(self.models)):

	  if self.monitor_targets:
	      X = data[i]
              Y = data[-1]
	  else:
	      X = data[i]
	      Y = None

	  model_data = (X, Y)
	  ch = self.models[i].get_monitoring_channels(model_data)
          for key in ch:
              value = ch[key]
              rval["cascade_" + str(i) + '_' + key] = value

          if Y is not None:
              state = self.fprop(data[0:-1])
              # Threshold Y_hat at 0.5.
              prediction = T.gt(state, 0.5)
              # If even one feature is wrong for a given training example,
              # it's considered incorrect, so we max over columns.
              incorrect = T.neq(Y, prediction).max(axis=1)
              rval['misclass'] = T.cast(incorrect, config.floatX).mean()
        return rval


    @wraps(Layer.get_params)
    def get_params(self):

	rlist = list()
	for model in self.models:
	  rlist =  rlist + model.get_params()
        return rlist


    ## FIXME
    @wraps(Layer.get_weight_decay) #check
    def get_weight_decay(self, coeffs, model_idx=-1):
	model = self.models[model_idx]
	rval = model.get_weight_decay(coeffs)

        return rval

    ## FIXME
    @wraps(Layer.get_l1_weight_decay) #check
    def get_l1_weight_decay(self, coeffs, model_idx=-1):
      	model = self.models[model_idx]
	rval = model.get_l1_weight_decay(coeffs)
        return rval


    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
	for model in self.models:
            model._modify_updates(updates)


    @wraps(Layer.get_lr_scalers) #check
    def get_lr_scalers(self, model_idx=-1):
        scaler = OrderedDict()
        for model in self.models:
            scaler.update(model.get_lr_scalers())
        return scaler


    @wraps(Layer.get_weights) #check
    def get_weights(self, model_idx=-1):

	model = self.models[model_idx]
        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")
        return model.layers[0].get_weights()


    @wraps(Layer.get_weights_view_shape) #check
    def get_weights_view_shape(self, model_idx = -1):

	model = self.models[model_idx]

        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")

        return model.layers[0].get_weights_view_shape()


    @wraps(Layer.get_weights_format) #check
    def get_weights_format(self, model_idx=-1):

	model = self.models[model_idx]
        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")
        return model.layers[0].get_weights_format()


    @wraps(Layer.get_weights_topo) #check
    def get_weights_topo(self, model_idx=-1):

	model = self.models[model_idx]
        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")

        return model.layers[0].get_weights_topo()


    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        import pdb
        pdb.set_trace()

        model_idx = -1
        return self.models[model_idx].layers[-1].cost(Y, Y_hat)


    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat): #check
        import pdb
        pdb.set_trace()
	model = self.models[-1]
        return model.layers[-1].cost_matrix(Y, Y_hat)


    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):
        import pdb
        pdb.set_trace()
	model = self.models[-1]
        return model.layers[-1].cost_from_cost_matrix(cost_matrix)


    def cost_from_X(self, data): #check
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an
        argument.

        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)

        Parameters
        ----------
        data : WRITEME
        """
        X, Y = data
        Y_hat = self.fprop(X) # is it consistent with dataset?

        ### Handle only binary label for now
        term_1 = -Y * T.log(Y_hat)
        term_2 = -(1 - Y) * T.log(1 - Y_hat)

        total = term_1 + term_2
        naxes = total.ndim
        axes_to_reduce = list(range(naxes))
        batch_axis = self.models[-1].layers[-1].output_space.get_batch_axis()
        del axes_to_reduce[batch_axis]
        ave = total.mean(axis=axes_to_reduce)
        #return self.cost(Y, Y_hat)
        return ave.mean()


    def get_monitoring_data_specs(self):
        """
        Returns data specs requiring both inputs and targets.

        Returns
        -------
        data_specs: TODO
            The data specifications for both inputs and targets.
        """

        model_conv = list()
        model_conv_items = list()

        for i in range(len(self.models)):
	  model_space = self.models[i].get_monitoring_data_specs()[0]
	  model_conv.append(model_space.components[0])

	  model_conv_items.append('features_' + str(i))

	model_conv.append(self.models[-1].get_monitoring_data_specs()[0].components[1])
	model_conv_items.append('targets')

        return (CompositeSpace(model_conv), tuple(model_conv_items))


    def get_input_source(self):
        """
        Returns a string, stating the source for the input. By default the
        model expects only one input source, which is called 'features'.
        """

        input_source = list()

        for i in range(len(self.models)):
	  model_space = self.models[i].get_monitoring_data_specs()[0]
	  input_source.append('features_' + str(i))

        return tuple(input_source)


    def get_target_source(self):
        """
        Returns a string, stating the source for the output. By default the
        model expects only one output source, which is called 'targets'.
        """

        return 'targets'


    def __str__(self):
        """
        Summarizes the cascade by printing the size and format of the input to all
        models.
        """
        rval = []
        for i in range(len(self.models)):
	    model_name = 'model ' + str(i) + ': '
            rval.append(model_name)
            input_space = model.get_input_space()
            rval.append('\tInput space: ' + str(input_space))
            rval.append('\tTotal input dimension: ' +
                        str(input_space.get_total_dimension()))
        rval = '\n'.join(rval)
        return rval


    @wraps(Layer.get_default_cost)
    def get_default_cost(self):
        return Default()
