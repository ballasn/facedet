import os, sys, getopt
from pylearn2.config import yaml_parse
from pylearn2 import train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.costs.cost import MethodCost
import exp.compression.hints_layerwise as hints
from utils.compression.TeacherDecayOverEpoch import TeacherDecayOverEpoch

def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    student_yaml = args[0]
  except getopt.GetoptError:
    usage()
    sys.exit(2) 
  
  #
  # TRAIN WITH TARGETS
  #

  # Load student
  with open(student_yaml, "r") as sty:
    student = yaml_parse.load(sty)
    
  # Remove teacher decay over epoch if there is one
  for ext in range(len(student.extensions)):
    if isinstance(student.extensions[ext],TeacherDecayOverEpoch):
      del student.extensions[ext]
  
  student.algorithm.cost = MethodCost(method='cost_from_X')

  # Change save paths
  for ext in range(len(student.extensions)):
    if isinstance(student.extensions[ext],MonitorBasedSaveBest):
      student.extensions[ext].save_path = student.save_path[0:-4] + "_noteacher_best.pkl"
  student.save_path = student.save_path[0:-4] + "_noteacher.pkl" 
  
  student.main_loop()
  
  #
  # TRAIN WITH TEACHER (TOP LAYER)
  #

  # Load student
  with open(student_yaml, "r") as sty:
    student = yaml_parse.load(sty)
    
  # Change save paths
  for ext in range(len(student.extensions)):
    if isinstance(student.extensions[ext],MonitorBasedSaveBest):
      student.extensions[ext].save_path = student.save_path[0:-4] + "_toplayer_best.pkl"
  student.save_path = student.save_path[0:-4] + "_toplayer.pkl" 
  
  student.main_loop()
  
  
  #
  # TRAIN WITH HINTS
  #
    
  hints.main([student_yaml, 'conv'])
    

  
if __name__ == "__main__":
  main(sys.argv[1:])