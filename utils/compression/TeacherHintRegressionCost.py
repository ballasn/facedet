import theano.tensor as T
import cPickle as pkl
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from utils.layer.convVariable import ConvElemwise
from theano.compat.python2x import OrderedDict

class TeacherHintRegressionCost(DefaultDataSpecsMixin, Cost):
    """
    Represents an objective function to be minimized by some
    `TrainingAlgorithm`.
    """
    
    # If True, the data argument to expr and get_gradients must be a
    # (X, Y) pair, and Y cannot be None.
    supervised = False
    
    def __init__(self, teacher_path, hintlayer):      
      # Load teacher network.
      fo = open(teacher_path, 'r')
      teacher = pkl.load(fo)
      fo.close()
      
      del teacher.layers[hintlayer+1:]

      self.teacher = teacher
      self.hintlayer = hintlayer

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
        x = data
	                    
        # Compute student output
        student_output = model.fprop(x)
        
        # Compute teacher output
        hint = x
        for l in range(self.hintlayer+1):
	  hint = self.teacher.layers[l].fprop(hint)
        
        # Change teacher format (1 vector of features instead of one feature map)
        hint = hint.reshape(shape=(hint.shape[0],
			           hint.shape[1]*
			           hint.shape[2]*
			           hint.shape[3]),ndim=2)

	# Compute cost
        #cost = -T.log(student_output) * hint 
        cost = 0.5*(hint - student_output)**2
        
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
               	
	rval = OrderedDict()
			
        value_cost_wrt_teacher = self.expr(model,data)

        if value_cost_wrt_teacher is not None:
	   name = 'cost_wrt_teacher'
	   rval[name] = value_cost_wrt_teacher
	   
        return rval        



        


