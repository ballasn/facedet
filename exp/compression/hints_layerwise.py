from pylearn2.config import yaml_parse
from pylearn2 import train
import math
import random
import os
import sys
import getopt
import numpy as np
import cPickle as pkl
from utils.compression.TeacherHintRegressionCost import TeacherHintRegressionCost
from utils.layer.convVariable import ConvElemwise
from utils.layer.SoftmaxBC01Extended import SoftmaxExtended
from pylearn2.models.mlp import Sigmoid, Softmax, RectifiedLinear, ConvRectifiedLinear, RectifierConvNonlinearity, SigmoidConvNonlinearity, TanhConvNonlinearity
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.space import VectorSpace
from copy import deepcopy

def splitStudentNetwork(student, fromto_student, teacher, hintlayer):
      
  # Check if we are in the softmax layers
  if isinstance(teacher.layers[hintlayer], Softmax) or isinstance(teacher.layers[hintlayer], SoftmaxExtended):
    assert (isinstance(student.model.layers[fromto_student[1]], Softmax) or isinstance(teacher.layers[hintlayer], SoftmaxExtended))
    assert teacher.layers[hintlayer].get_output_space().dim == student.model.layers[fromto_student[1]].get_output_space().dim
    
  else:
    # Retrieve student subnetwork
    if fromto_student[1] < len(student.model.layers)-1:
      del student.model.layers[fromto_student[1]+1:] 
    
    # Add regressor if hint layer has more outputs than corresponding student layer
    if teacher.layers[hintlayer].get_output_space().num_channels > student.model.layers[fromto_student[1]].get_output_space().num_channels:
      dim = teacher.layers[hintlayer].output_space.get_total_dimension()
      layer_name = 'hint_regressor'
      if isinstance(teacher.layers[hintlayer], MaxoutConvC01B):
	hint_reg_layer = Maxout(layer_name, dim, 2, irange= .005, max_col_norm= 1.9365)
      elif isinstance(teacher.layers[hintlayer], ConvRectifiedLinear):
	hint_reg_layer = RectifiedLinear(dim=dim, layer_name=layer_name, irange=0.05)
      elif isinstance(teacher.layers[hintlayer], ConvElemwise):
	if isinstance(teacher.layers[hintlayer].nonlinearity,RectifierConvNonlinearity):
	  hint_reg_layer = RectifiedLinear(dim=dim, layer_name=layer_name, irange=0.05)
	elif isinstance(teacher.layers[hintlayer].nonlinearity,SigmoidConvNonlinearity) or isinstance(teacher.layers[hintlayer].nonlinearity,TanhConvNonlinearity):
	  hint_reg_layer = Sigmoid(dim=dim, layer_name=layer_name, irange=0.05)
	else:
	  raise AssertionError("Unknown layer type")
      else:
	raise AssertionError("Unknown layer type")
      
    # Include regressor layer in student subnetwork  
    hint_reg_layer.set_mlp(student.model)  
    hint_reg_layer.set_input_space(student.model.layers[-1].output_space)
    student.model.layers.append(hint_reg_layer)

    # Change cost to optimize wrt teacher hints
    student.algorithm.cost = TeacherHintRegressionCost(teacher,hintlayer)
    
    # Set monitor_targets to false
    student.model.monitor_targets = False

    # Change monitored channel
    student.extensions[0].channel_name = "valid_cost_wrt_teacher"
    student.algorithm.termination_criterion.channel_name = "valid_cost_wrt_teacher"
    student.algorithm.termination_criterion._channel_name = "valid_cost_wrt_teacher"
  
  # Change save paths
  student.save_path = student.extensions[0].save_path[0:-4] + "_hintlayer" + str(fromto_student[1]) + ".pkl"
  student.extensions[0].save_path = student.save_path[0:-4] + "_best.pkl"
    
  # Freeze parameters of the layers trained in the last subnetworks
  for i in range(0,fromto_student[0]-1):
    student.model.freeze(student.model.layers[i].get_params())

  return student
    
def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    teacher_pkl = args[0] 
    student_yaml = args[1]
  except getopt.GetoptError:
    usage()
    sys.exit(2) 

  student_savepath = './models/student_nets/'
  
  if not os.path.exists(student_savepath):
    os.makedirs(student_savepath)
  
  # Layers correspondance (hints)
  #student_layers = [2,5,7]
  #teacher_layers = [0,2,4]
  
  student_layers = [2,4] 
  teacher_layers = [0,2]
  #[[0,2],[2,4]]

  # Load student
  with open(student_yaml, "r") as sty:
    student = yaml_parse.load(sty)
    
  # Load teacher network
  fo = open(teacher_pkl, 'r')
  teacher = pkl.load(fo)
  fo.close()
  
  assert len(student_layers) == len(teacher_layers)
  n_hints = len(student_layers)
  assert max(student_layers) <= len(student.model.layers)-2
  if isinstance(teacher.layers[-1], SoftmaxExtended):
    assert max(teacher_layers) <= len(teacher.layers)-3
  else:
    assert max(teacher_layers) <= len(teacher.layers)-2
  
  # Train layers with teacher hints 
  for i in range(n_hints):
    print 'Training student hint layer %d out of %d' % (i+1, n_hints)
      
    # Select student block of layers forming subnetwork
    bottom_layer = student_layers[i-1]+1 if i>0 else 0
    top_layer = student_layers[i]
    
    # Copy student and teacher to be able to modify them
    teacher_aux = deepcopy(teacher)
    student_aux = deepcopy(student)

    # Retrieve student subnetwork and add regression to teacher layer
    student_hint = splitStudentNetwork(student_aux, [bottom_layer, top_layer], teacher_aux, teacher_layers[i])
    
    # Train student subnetwork
    student_hint.main_loop()
      
    # Save complete student subnetwork
    hint_output = open(student_savepath + 'student_subnetwork' + str(i) + '.pkl', 'wb')
    pkl.dump(student_hint, hint_output)
    hint_output.close()
      
    # Save pretrained student subnetworks together (without regression to teacher layer)
    student.model.layers[0:top_layer] = student_hint.model.layers[0:-2]

  print 'Training student softmax layer'

  # Train softmax layer and stack it to the pretrained student network
  softmax_hint = splitStudentNetwork(student, [len(student.model.layers)-1, len(student.model.layers)-1], teacher, len(teacher.layers)-1)  
  softmax_hint.main_loop()
  student.model.layers[-1] = softmax_hint.model.layers[-1]
     
  # Save pretrained student network to pkl file
  student_final = open(student_savepath + 'student_complete.pkl', 'wb')
  pkl.dump(student,student_final)
  student_final.close()
      
  # TODO: Finetune student network and save it
    
  
if __name__ == "__main__":
  main(sys.argv[1:])
