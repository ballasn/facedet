import theano.tensor as T
import cPickle as pkl
from theano.compat.python2x import OrderedDict
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost

class TeacherRegressionCost(DefaultDataSpecsMixin, Cost):
    """
    Represents an objective function to be minimized by some
    `TrainingAlgorithm`.
    """
    
    # If True, the data argument to expr and get_gradients must be a
    # (X, Y) pair, and Y cannot be None.
    supervised = True
    
    def __init__(self, teacher_path, relaxation_term=1, weight=1):
      self.relaxation_term = relaxation_term
      
      # Load teacher network and change parameters according to relaxation_term.
      print teacher_path
      fo = open(teacher_path, 'r')
      teacher = pkl.load(fo)
      fo.close()
      
      tparams = teacher.layers[-1].get_param_values()
      tparams = [x/float(self.relaxation_term) for x in tparams]
      teacher.layers[-1].set_param_values(tparams)

      self.teacher = teacher
      self.weight = weight
      
    def cost_wrt_target(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (x, y) = data
                
        targets = T.argmax(y, axis=1)
                        
        # Compute student output
        Ps_y_given_x = model.fprop(x)
        
        # Compute cost
        rval = -T.log(Ps_y_given_x)[T.arange(targets.shape[0]), targets]
        
        return rval
        
    def cost_wrt_teacher(self, model, data):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (x, y) = data
                                                
        # Compute teacher relaxed output
	Pt_y_given_x_relaxed = self.teacher.fprop(x)

	# Relax student softmax layer using relaxation_term.
	sparams = model.layers[-1].get_param_values()
	sparams = [item/float(self.relaxation_term) for item in sparams]
	model.layers[-1].set_param_values(sparams)
	        
        # Compute student relaxed output
        Ps_y_given_x_relaxed = model.fprop(x)

	# Compute cost
        rval = -T.log(Ps_y_given_x_relaxed) * Pt_y_given_x_relaxed 
        rval = T.mean(rval, axis=1)
        
        return rval  
        
    def expr(self, model, data, ** kwargs):
        """
        Returns a theano expression for the cost function.
        
        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments. Not used by the base class.
        """
	cost_wrt_y = self.cost_wrt_target(model,data)
        cost_wrt_teacher = self.cost_wrt_teacher(model,data)
        
	# Compute cost
        cost = self.weight*cost_wrt_y + cost_wrt_teacher
        
        return T.mean(cost)
        
    def get_monitoring_channels(self, model, data, **kwargs):
        """
        .. todo::

            WRITEME

        .. todo::

            how do you do prereqs in this setup? (I think PL changed
            it, not sure if there still is a way in this context)

        Returns a dictionary mapping channel names to expressions for
        channel values.

        Parameters
        ----------
        model : Model
            the model to use to compute the monitoring channels
        data : batch
            (a member of self.get_data_specs()[0])
            symbolic expressions for the monitoring data
        kwargs : dict
            used so that custom algorithms can use extra variables
            for monitoring.

        Returns
        -------
        rval : dict
            Maps channels names to expressions for channel values.
        """
               
	rval = super(TeacherRegressionCost, self).get_monitoring_channels(model,data)
	                
        value_cost_wrt_target = self.cost_wrt_target(model,data)
        if value_cost_wrt_target is not None:
	   name = 'cost_wrt_target'
	   rval[name] = self.weight*T.mean(value_cost_wrt_target)
                
        value_cost_wrt_teacher = self.cost_wrt_teacher(model,data)
        if value_cost_wrt_teacher is not None:
	   name = 'cost_wrt_teacher'
	   rval[name] = T.mean(value_cost_wrt_teacher)
	   	
        return rval        








