import os
import sys
import getopt

from pylearn2.config import yaml_parse
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.models.mlp import Softmax

def numberMult(model):
  
  mult = 0
  
  previous_output = model.model.input_space.num_channels
  
  for i in range(0,len(model.model.layers)):
    print i
    
    if isinstance(model.model.layers[i], MaxoutConvC01B):
      detector = model.model.layers[i].detector_space.shape[0]*model.model.layers[i].detector_space.shape[1]
      kernel = model.model.layers[i].kernel_shape[0]*model.model.layers[i].kernel_shape[1]
      
      mult = mult + detector*kernel*previous_output*model.model.layers[i].output_space.num_channels
      previous_output = model.model.layers[i].output_space.num_channels
    elif isinstance(model.model.layers[i], Maxout):
      if isinstance(model.model.layers[i-1], MaxoutConvC01B):
	input_space = model.model.layers[i].input_space.shape[0]*model.model.layers[i].input_space.shape[1]
	mult = mult + input_space*model.model.layers[i].input_space.num_channels*model.model.layers[i].output_space.dim
      else:
	mult = mult + input_space*model.model.layers[i].input_space.dim*model.model.layers[i].output_space.dim
    elif isinstance(model.model.layers[i], Softmax):
      if isinstance(model.model.layers[i-1], MaxoutConvC01B):
	input_space = model.model.layers[i].input_space.shape[0]*model.model.layers[i].input_space.shape[1]
	mult = 2*mult + input_space*model.model.layers[i].input_space.num_channels*model.model.layers[i].output_space.dim
      else:
	mult = 2*mult + model.model.layers[i].input_space.dim*model.model.layers[i].output_space.dim
    else:
      print 'error'
          
  return mult
      
def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    model_yaml = args[0]
  except getopt.GetoptError:
    usage()
    sys.exit(2) 


  # Load student
  with open(model_yaml, "r") as sty:
    model = yaml_parse.load(sty)
  
  result = numberMult(model)
    
  print 'Number of multiplications is %is' % (result)

  
if __name__ == "__main__":
  main(sys.argv[1:])
