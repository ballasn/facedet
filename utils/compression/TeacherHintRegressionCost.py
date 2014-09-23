import theano.tensor as T
import cPickle as pkl
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from utils.layer.convVariable import ConvElemwise

class TeacherHintRegressionCost(DefaultDataSpecsMixin, Cost):
    """
    Represents an objective function to be minimized by some
    `TrainingAlgorithm`.
    """
    
    # If True, the data argument to expr and get_gradients must be a
    # (X, Y) pair, and Y cannot be None.
    supervised = True
    
    def __init__(self, teacher_path, hintlayer):      
      # Load teacher network.
      fo = open(teacher_path, 'r')
      teacher = pkl.load(fo)
      fo.close()
      
      del teacher.layers[hintlayer+1:]
      teacher.set_input_space(teacher.layers[0].input_space)
      
      print 'teacher space: ' 
      print teacher.get_output_space()


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
        (x, y) = data
	                    
        # Compute student output
        student_output = model.fprop(x)
        
        # Compute teacher output
        hint = x
        for l in range(self.hintlayer+1):
	  hint = model.layers[l].fprop(hint)
	
	# Check teacher nonlinearity and modify output if necessary
        if isinstance(model.layers[self.hintlayer], ConvElemwise) and teacher.layers[self.hintlayer].nonlin is 'Tanh':
	  hint = (hint + 1)/float(2)
        
        # Change teacher format (1 vector of features instead of one feature map)
	hint.flatten()
	
	# Compute cost
        cost = -T.log(student_output) * hint 
        
        return T.mean(cost)


