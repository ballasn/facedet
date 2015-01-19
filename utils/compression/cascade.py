import logging
import math
import operator
import sys
import warnings

from theano.ifelse import ifelse
from pylearn2.utils import wraps
from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace
# Add cascade cost

class Cascade(object):

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
        #super(Cascade, self).__init__(**kwargs)
        
        if thresholds is not None:
	  assert(len(models) == len(thresholds) - 1)
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
      
      
        if not hasattr(self.models[0], "input_space"):
            raise AttributeError("Input space has not been provided for model 0.")
	  
        rval = self.models[0].fprop(state_below(0)) # Is state_below a tuple?
      
	rlist = [rval]

        for i in range(len(1,self.models[1:])):
	  if rval > self.thresholds[i-1]:
	    self.models[i].fprop(state_below(i)) # Is state_below a tuple?
	    rlist.append(rval)
	    
	    #rval = theano.ifelse(rval > self.thresholds[i-1], self.models[i].fprop(state_below[i]), 0)
            

        if return_all:
            return rlist
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
        

    def freeze(self, parameter_set, model_idx):
        """
        Freezes some of the parameters (new theano functions that implement
        learning will not use them; existing theano functions will continue
        to modify them).

        Parameters
        ----------
        parameter_set : set
            Set of parameters to freeze.
        """

        self.models[model_idx].freeze_set = self.models[model_idx].freeze_set.union(parameter_set)
        

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
      
	rlist = list()
        
        for i in range(len(self.models)):
	  
	  if self.monitor_targets:
	      X, Y = data
	  else:
	      X = data
	      Y = None
	      
	  model_data.X = X(i) # Check consistency with dataset
	  model_data.Y = Y
	      
	  rval = model.get_monitoring_channels(model_data)
	  rlist.append(rval)
	      
        return rlist
        
        
    @wraps(Layer.get_params)
    def get_params(self):
      
	rlist = list()
	
	for model in self.models:
	  rlist =  rlist + model.get_params()

        return rlist


    @wraps(Layer.get_weight_decay) #check
    def get_weight_decay(self, coeffs, model_idx):
      
	model = self.models[model_idx]
	rval = model.get_weight_decay(coeffs)

        return rval
        

    @wraps(Layer.get_l1_weight_decay) #check
    def get_l1_weight_decay(self, coeffs, model_idx):
      
      	model = self.models[model_idx]
	rval = model.get_l1_weight_decay(coeffs)

        return rval


    @wraps(Layer._modify_updates) #check
    def _modify_updates(self, updates, model_idx):

        model = self.models[model_idx]
        model._modify_updates(updates)


    @wraps(Layer.get_lr_scalers) #check
    def get_lr_scalers(self, model_idx):
      
        model = self.models[model_idx]

        return model.get_lr_scalers_from_layers(model)


    @wraps(Layer.get_weights) #check
    def get_weights(self, model_idx):

	model = self.models[model_idx]

        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")

        return model.layers[0].get_weights()


    @wraps(Layer.get_weights_view_shape) #check
    def get_weights_view_shape(self, model_idx):

	model = self.models[model_idx]
	
        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")

        return model.layers[0].get_weights_view_shape()
        

    @wraps(Layer.get_weights_format) #check
    def get_weights_format(self, model_idx):
      
	model = self.models[model_idx]

        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")

        return model.layers[0].get_weights_format()
        

    @wraps(Layer.get_weights_topo) #check
    def get_weights_topo(self, model_idx):
      
	model = self.models[model_idx]

        if not hasattr(model, "input_space"):
            raise AttributeError("Input space has not been provided.")

        return model.layers[0].get_weights_topo() 


    @wraps(Layer.cost) #check
    def cost(self, Y, Y_hat, model_idx):
      
	model = self.models[model_idx]

        return model.layers[-1].cost(Y, Y_hat)


    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat, model_idx): #check

	model = self.models[model_idx]
	
        return model.layers[-1].cost_matrix(Y, Y_hat)


    @wraps(Layer.cost_from_cost_matrix) #check
    def cost_from_cost_matrix(self, cost_matrix, model_idx):

	model = self.models[model_idx]
        return model.layers[-1].cost_from_cost_matrix(cost_matrix)


    def cost_from_X(self, data, model_idx): #check
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an
        argument.

        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)

        Parameters
        ----------
        data : WRITEME
        """
        model = self.models[model_idx]
        
        model.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hat = self.fprop(X(model_idx)) # is it consistent with dataset?
        return self.cost(Y, Y_hat)
        
        
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
        
 
    @wraps(Layer.get_default_cost)
    def get_default_cost(self):

        return 0 #FIXME: cascade cost as default cost
        
      

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
        
        
        
 