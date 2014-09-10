import theano.tensor as T
import cPickle as pkl
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost

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
      fo = open(teacher_path, 'r')
      teacher = pkl.load(fo)
      fo.close()
      
      tparams = teacher.layers[-1].get_param_values()
      tparams = [x/float(self.relaxation_term) for x in tparams]
      teacher.layers[-1].set_param_values(tparams)

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
        
        targets = T.argmax(y, axis=1)
                
        # Compute student output
        Ps_y_given_x = model.fprop(x)
        
        # Compute teacher relaxed output
	Pt_y_given_x_relaxed = self.teacher.fprop(x)

	# Relax student softmax layer using relaxation_term.
	sparams = model.layers[-1].get_param_values()
	sparams = [item/float(self.relaxation_term) for item in sparams]
	model.layers[-1].set_param_values(sparams)
	        
        # Compute student relaxed output
        Ps_y_given_x_relaxed = model.fprop(x)
	
	# Compute cost
        cost_wrt_y = -T.log(Ps_y_given_x)[T.arange(targets.shape[0]), targets]
        cost_wrt_teacher = -T.log(Ps_y_given_x_relaxed) * Pt_y_given_x_relaxed 
        cost = (1/float(self.relaxation_term))*cost_wrt_y + T.mean(cost_wrt_teacher, axis=1)
        
        return T.mean(cost)




