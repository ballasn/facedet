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
from pylearn2.models.mlp import Sigmoid, Tanh, RectifiedLinear, ConvRectifiedLinear
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.space import VectorSpace

def splitStudentNetwork(student, dataset, fromto_student, teacher_pkl, hintlayer):
  
  nl_student = len(student.model.layers)

  # Retrieve student subnetwork
  del student.model.layers[0:fromto_student[0]]
  del student.model.layers[fromto_student[1]+1:] 

  # Load teacher network
  fo = open(teacher_pkl, 'r')
  teacher = pkl.load(fo)
  fo.close()
  
  # Change student input data
  student.dataset = dataset
  nl_student = len(student.model.layers)
  
  student.dataset = dataset  
    
  # Add regressor layer if number of outputs in hint layer > number of outputs in corresponding student layer
  if teacher.layers[hintlayer].output_channels > student.model.layers[fromto_student[1]].output_channels:
    dim = teacher.layers[hintlayer].output_space.get_total_dimension()
    layer_name = 'hint_regressor'
    
    if isinstance(teacher.layers[hintlayer], MaxoutConvC01B):
      raise NotImplementedError("Teacher hints not implemented for maxout layers")
      #hint_reg_layer = Maxout(layer_name, dim, 2, irange= .005, max_col_norm= 1.9365)
    elif isinstance(teacher.layers[hintlayer], ConvRectifiedLinear):
      hint_reg_layer = RectifiedLinear(dim=dim, layer_name=layer_name, irange=0.05,sparse_init=None)
    elif isinstance(teacher.layers[hintlayer], ConvElemwise):
      if teacher.layers[hintlayer].nonlin is 'Sigmoid' or teacher.layers[hintlayer].nonlin is 'Tanh':
	hint_reg_layer = Sigmoid(dim=dim, layer_name=layer_name, irange=0.05,sparse_init=None)
      elif teacher.layers[hintlayer].nonlin is 'RectifiedLinear':
	hint_reg_layer = RectifiedLinear(dim=dim, layer_name=layer_name, irange=0.05,sparse_init=None)
    else:
      raise AssertionError("Unknown layer type")
      
    hint_reg_layer.set_mlp(student.model)  
    hint_reg_layer.set_input_space(student.model.layers[-1].output_space)
    student.model.layers.append(hint_reg_layer)
    
  # Set monitor_targets to false
  student.model.monitor_targets = False

  # Change monitored channel
  student.extensions[0].channel_name = "valid_cost_wrt_teacher"
  student.algorithm.termination_criterion.channel_name = "valid_cost_wrt_teacher"
  student.algorithm.termination_criterion._channel_name = "valid_cost_wrt_teacher"
  
  # Change save paths
  student.save_path = student.extensions[0].save_path[0:-4] + "_hintlayer" + str(fromto_student[-1]) + ".pkl"
  student.extensions[0].save_path = student.save_path[0:-4] + "_best.pkl"

  # Change cost to optimize wrt teacher hints
  if fromto_student[-1] < nl_student:
    student.algorithm.cost = TeacherHintRegressionCost(teacher_pkl,hintlayer)

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
  
  # Layers correspondance (hints)
  student_layers = [3]
  teacher_layers = [0]

  # Load student
  with open(student_yaml, "r") as sty:
    student = yaml_parse.load(sty)
          
  for i in range(len(student_layers)):
    # Select student block of layers forming subnetwork
    bottom_layer = student_layers[i-1] if i>0 else 0
    top_layer = student_layers[i]

    # Retrieve student subnetwork and add regression to teacher layer
    student_hint = splitStudentNetwork(student, student.dataset, [bottom_layer, top_layer], teacher_pkl, teacher_layers[i])
                 
    # Train student subnetwork
    student_hint.main_loop()
    
    # Save complete student subnetwork
    hint_output = open(student_savepath + 'student_subnetwork' + str(i) + '.pkl', 'wb')
    pkl.dump(student_hint, hint_output)
    hint_output.close()
   
    # Stack student subnetworks together (avoid regression to teacher layer)
    student.model.layers[bottom_layer, top_layer] = student_hint.model.layers[0:-2]
          
  # Save trained student network to pkl file
  student_final = open(student_savepath + 'student_complete.pkl', 'wb')
  pkl.dump(student_final, student)
  student_final.close()
  
  # TODO: Finetune student network and save it
  student_final_postfinetuning = open(student_savepath + 'student_complete_postfinetuning.pkl', 'wb')
  pkl.dump(student_final_finetuning, student)
  student_final_postfinetuning.close()
  
  
if __name__ == "__main__":
  main(sys.argv[1:])
