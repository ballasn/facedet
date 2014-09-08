import theano.tensor as T
import cPickle as pkl
from pylearn2.utils.data_specs import DataSpecsMapping, Cost

class TeacherRegressionCost(DefaultDataSpecsMixin, Cost):
    """
    Represents an objective function to be minimized by some
    `TrainingAlgorithm`.
    """
    
    # If True, the data argument to expr and get_gradients must be a
    # (X, Y) pair, and Y cannot be None.
    supervised = True
    
    def __init__(self, teacher_path, relaxation_term=1):
      self.relaxation_term = relaxation_term
      
      # Load teacher network and change parameters according to relaxation_term.
      with open(teacher_path, 'r') as t_p:
	teacher = pkl.load(t_p)

	teacher.layers[-1].set_param_values(teacher.layers[-1].get_param_values()
					  / float(self.relaxation_term))
					  
	self.teacher = teacher

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
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (x, y) = data
        
        # Compute student output
        Ps_y_given_x = model.fprop(x)
        
        # Compute teacher relaxed output
	Pt_y_given_x_relaxed = self.teacher.fprop(x)

        # Relax student softmax layer using relaxation_term.
        model.layers[-1].set_param_values(model.layers[-1].get_param_values()
					  / float(self.relaxation_term))
        
        # Compute student relaxed output
        Ps_y_given_x_relaxed = model.fprop(x)

        
        cost_wrt_y = -T.log(Ps_y_given_x)[T.arange(y.shape[0]), y]
        cost_wrt_teacher = -T.log(Ps_y_given_x_relaxed) * Pt_y_given_x_relaxed 
        
        return cost_wrt_y + T.mean(cost_wrt_teacher, axis=1)

    def get_gradients(self, model, data, ** kwargs):
        """
        Provides the gradients of the cost function with respect to the model
        parameters.

        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments, not used by the base class.

        Returns
        -------
        gradients : OrderedDict
            a dictionary mapping from the model's parameters
            to their gradients
            The default implementation is to compute the gradients
            using T.grad applied to the value returned by expr.
            However, subclasses may return other values for the gradient.
            For example, an intractable cost may return a sampling-based
            approximation to its gradient.
        updates : OrderedDict
            a dictionary mapping shared variables to updates that must
            be applied to them each time these gradients are computed.
            This is to facilitate computation of sampling-based approximate
            gradients.
            The parameters should never appear in the updates dictionary.
            This would imply that computing their gradient changes
            their value, thus making the gradient value outdated.
        """

        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError:
            message = "Error while calling " + str(type(self)) + ".expr"
            reraise_as(TypeError(message))

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs='ignore')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates

    def get_monitoring_channels(self, model, data, **kwargs):
        """
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
        self.get_data_specs(model)[0].validate(data)
        return OrderedDict()


