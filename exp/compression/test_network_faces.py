from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
from time import time
import theano.tensor as T
from theano import function
from utils.cascade.cascadeFDDB import process_fold
import cPickle as pkl
from math import sqrt


def compute_test_accuracy(model, out_dir, nfolds):
    sizes = [16]
    strides = [2]
    ratio = sqrt(2)
    global_scales = [0.05 * ratio**e for e in range(5)]
    local_scales = [global_scales]
    base_size = max(sizes)

    probs = [0.1]

    # Compile functions
    x = T.tensor4('x')
    predict = function([x], model.fprop(x))

    models = [model]
    fprops = [predict]

    t_orig = time()
    for nb in range(1, nfolds+1):
        t0 = time()
        process_fold(models, fprops, local_scales, sizes, strides, probs,
                     nb, out_dir, mode='rect')
        t = time()
        print ""
        print t-t0, 'seconds for the fold'
    print t-t_orig, 'seconds for FDDB'
    
if __name__ == '__main__':
  
  if len(sys.argv) != 4:
    raise AssertionError('Wrong number of input arguments: teacher_path, student_path, out_dir')

  # Parse arguments
  _, teacher_path, student_path, out_dir = sys.argv
  
  nfolds = 10
  
  # Load student model
  with open(teacher_path, 'r') as s_p:
    teacher = pkl.load(s_p)
  
  # Evaluate teacher
  start_time_teacher = time()
  acc_teacher = compute_test_accuracy(teacher, out_dir+'_teacher',nfolds)
  elapsed_time_teacher = time() - start_time_teacher
  
  # Load student model
  with open(student_path, 'r') as s_p:
    student = pkl.load(s_p)

  # Evaluate student
  start_time_student = time()
  acc_student = compute_test_accuracy(student, out_dir+'_student',nfolds)
  elapsed_time_student = time() - start_time_student

  # Evaluate performance/time  
  time_inc = elapsed_time_student - elapsed_time_teacher

  # Print results
  print 'Teacher time is %fs' % (elapsed_time_teacher)
  print 'Student time is %fs' % (elapsed_time_student)
  print 'Error increment is ... and time increment is %fs' % (time_inc)
