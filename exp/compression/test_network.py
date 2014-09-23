from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import theano.tensor as T
from theano import function
import time


# Parse arguments
_, teacher_path, student_path = sys.argv

# Set variables
batch_size = 100

# Load teacher model
teacher = serial.load(teacher_path)
teacher.set_batch_size(batch_size)

# Load student model
student = serial.load(student_path)
student.set_batch_size(batch_size)

# Load dataset
src = teacher.dataset_yaml_src
assert src.find('train') != -1
test = yaml_parse.load(src)
test = test.get_test_set()
test.X = test.X.astype('float32')

assert test.X.shape[0] % batch_size == 0

def compute_test_accuracy(model):
    test_acc = []
    
    Xb = model.get_input_space().make_batch_theano()
    Xb.name = 'Xb'
    yb = model.get_output_space().make_batch_theano()
    yb.name = 'yb'
    
    y_model = model.fprop(Xb)
    label = T.argmax(yb,axis=1)
    prediction = T.argmax(y_model,axis=1)
    acc_model = 1.-T.neq(label , prediction).mean()
    
    batch_acc = function([Xb,yb],[acc_model])
        
    iterator = test.iterator(mode = 'even_sequential',
                            batch_size = batch_size,
                            data_specs = model.cost_from_X_data_specs())
                            
    for item in iterator:
        x_arg, y_arg = item	  
        test_acc.append(batch_acc(x_arg, y_arg)[0])
    return sum(test_acc) / float(len(test_acc))

# Evaluate teacher
start_time_teacher = time.time()
acc_teacher = compute_test_accuracy(teacher)
elapsed_time_teacher = time.time() - start_time_teacher
error_teacher = 1. - acc_teacher 

# Evaluate student
start_time_student = time.time()
acc_student = compute_test_accuracy(student)
elapsed_time_student = time.time() - start_time_student
error_student = 1. - acc_student 

# Evaluate performance/time
error_inc = error_student - error_teacher
time_inc = elapsed_time_student - elapsed_time_teacher

# Print results
print 'Teacher achieved %f error in %fs' % (error_teacher, elapsed_time_teacher)
print 'Student achieved %f error in %fs' % (error_student, elapsed_time_student)
print 'Error increment is %f and time increment is %fs' % (error_inc, time_inc)

