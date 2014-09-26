import cPickle as pkl
import numpy as np
import sys
import theano.tensor as T
from theano import function
# Custom imports

if len(sys.argv) != 3:
    print "Usage : testExtended.py <model_file> <image_size>"
    sys.exit(2)

# Define the model
model_file = sys.argv[1]
im_size = int(sys.argv[2])

with open(model_file, 'r') as m_f:
    model = pkl.load(m_f)

# Compile network prediction function
x = T.tensor4("x")
#model.layers = model.layers[:-1]
fprop_func = function([x], model.fprop(x))
print fprop_func

mini_batch = np.random.uniform(0, 256, size=(1, im_size, im_size, 3))
mini_batch = np.asarray(mini_batch, dtype=np.float32)
mini_batch = np.transpose(mini_batch, (3, 1, 2, 0))

print "shape of the minibatch", mini_batch.shape
results = fprop_func(mini_batch)
print results[0,:,:,0]
print "shape of the results", results.shape

mini_batch = np.random.uniform(0, 256, size=(1, 2*im_size, 2*im_size, 3))
mini_batch = np.asarray(mini_batch, dtype=np.float32)
mini_batch = np.transpose(mini_batch, (3, 1, 2, 0))

print "shape of the minibatch", mini_batch.shape
results = fprop_func(mini_batch)
print results[0,:,:,0]
print "shape of the results", results.shape

