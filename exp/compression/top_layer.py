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
from models.layer.convVariable import ConvElemwise
from models.layer.SoftmaxBC01Extended import SoftmaxExtended
from models.layer.SigmoidBC01Extended import SigmoidExtended
from pylearn2.models.mlp import Sigmoid, Softmax, RectifiedLinear, ConvRectifiedLinear, RectifierConvNonlinearity, SigmoidConvNonlinearity, TanhConvNonlinearity
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.space import VectorSpace
from copy import deepcopy

    
def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    student_yaml = args[0]
    load_layer = int(args[1])
  except getopt.GetoptError:
    usage()
    sys.exit(2) 

  # Load student
  with open(student_yaml, "r") as sty:
    student = yaml_parse.load(sty)
    
  # Load hints
  if student.algorithm.cost.hints is not None:
    student_layers = list(zip(*student.algorithm.cost.hints)[0]) 
    teacher_layers = list(zip(*student.algorithm.cost.hints)[1])
    
    n_hints = len(student_layers)

  else:
    n_hints = 0
  
  # Load pretrained student network
  fo = open(student.extensions[0].save_path[0:-4] + "_hintlayer" + str(load_layer) + "_best.pkl", 'r')
  pretrained_model = pkl.load(fo)
  fo.close()
  
  student.model.layers[0:load_layer+1] = pretrained_model.layers[0:load_layer+1]  
  
  student.save_path = student.extensions[0].save_path[0:-4] + "_hint" + str(load_layer) + "_softmax.pkl"
  student.extensions[0].save_path = student.save_path[0:-4] + "_best.pkl"

  student.main_loop()
  
  
if __name__ == "__main__":
  main(sys.argv[1:])
