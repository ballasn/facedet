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
        
        Ps_y_given_x = Ps_y_given_x.reshape(shape=(Ps_y_given_x.shape[0],
                       Ps_y_given_x.shape[1]*
                       Ps_y_given_x.shape[2]*
                       Ps_y_given_x.shape[3]),ndim=2)
        
        # Compute teacher relaxed output
	Pt_y_given_x_relaxed = self.teacher.fprop(x)
        Pt_y_given_x_relaxed = Pt_y_given_x_relaxed.reshape(shape=(Pt_y_given_x_relaxed.shape[0],
			       Pt_y_given_x_relaxed.shape[1]*
			       Pt_y_given_x_relaxed.shape[2]*
			       Pt_y_given_x_relaxed.shape[3]),ndim=2)	
	

	# Relax student softmax layer using relaxation_term.
	sparams = model.layers[-1].get_param_values()
	sparams = [item/float(self.relaxation_term) for item in sparams]
	model.layers[-1].set_param_values(sparams)
	        
        # Compute student relaxed output
        Ps_y_given_x_relaxed = model.fprop(x)
        
        Ps_y_given_x_relaxed = Ps_y_given_x_relaxed.reshape(shape=(Ps_y_given_x_relaxed.shape[0],
			       Ps_y_given_x_relaxed.shape[1]*
			       Ps_y_given_x_relaxed.shape[2]*
			       Ps_y_given_x_relaxed.shape[3]),ndim=2)	
                
	# Compute cost
        cost_wrt_y = -T.log(Ps_y_given_x)[T.arange(targets.shape[0]), targets]
        cost_wrt_teacher = -T.log(Ps_y_given_x_relaxed) * Pt_y_given_x_relaxed 
        #cost = T.mean(cost_wrt_teacher)
        cost = self.weight*cost_wrt_y + T.mean(cost_wrt_teacher, axis=1)
        
        return T.mean(cost)




