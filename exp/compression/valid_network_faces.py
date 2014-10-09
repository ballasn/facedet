from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
from time import time
import theano.tensor as T
from theano import function
import cPickle as pkl
from math import sqrt
from datasets.faceDataset import faceDataset, FaceIterator
import numpy as np



  
def compute_test_accuracy(model, iterator, nb):
    test_acc = []
    
    Xb = model.get_input_space().make_batch_theano()
    Xb.name = 'Xb'
    yb = model.get_output_space().make_batch_theano()
    yb.name = 'yb'
    
    y_model = model.fprop(Xb)
    label = T.argmax(yb,axis=1)
    prediction = T.argmax(y_model,axis=1)
    acc_model = 1.-T.neq(label , prediction.flatten(ndim=1)).mean()
    
    batch_acc = function([Xb,yb],[acc_model])
    
    iterator = valid.iterator(mode = 'even_sequential',
                            batch_size = nb,
                            data_specs = model.cost_from_X_data_specs())

    for item in iterator:
        x_arg, y_arg = item
        test_acc.append(batch_acc(x_arg, y_arg)[0])
    return sum(test_acc) / float(len(test_acc))

    
if __name__ == '__main__':
  
  if len(sys.argv) != 3:
    raise AssertionError('Wrong number of input arguments: teacher_path, student_path')

  # Parse arguments
  _, teacher_path, student_path = sys.argv
  
  nb = 128
    
  # Load teacher model
  with open(teacher_path, 'r') as s_p:
    teacher = pkl.load(s_p)
  
  # Load student model
  with open(student_path, 'r') as s_p:
    student = pkl.load(s_p)

  # Load dataset
  src = student.monitor._datasets[1]
  valid = yaml_parse.load(src)

  
  #fi = FaceIterator(dataset=valid, batch_size=128)

  # Evaluate teacher
  start_time_teacher = time()
  acc_teacher = compute_test_accuracy(teacher, valid, nb)
  elapsed_time_teacher = time() - start_time_teacher
  error_teacher = 1. - acc_teacher 
  
  # Evaluate student
  start_time_student = time()
  acc_student = compute_test_accuracy(student, valid, nb)
  elapsed_time_student = time() - start_time_student
  error_student = 1. - acc_student 

  # Evaluate performance/time  
  error_inc = error_student - error_teacher
  time_inc = elapsed_time_student - elapsed_time_teacher

  # Print results
  print 'Teacher time is %fs' % (elapsed_time_teacher)
  print 'Student time is %fs' % (elapsed_time_student)
  print 'Teacher error is %fs' % (error_teacher)
  print 'Student error is %fs' % (error_student)
  print 'Error increment is %f and time increment is %fs' % (error_inc, time_inc)
