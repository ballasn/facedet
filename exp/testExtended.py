import cPickle as pkl
import numpy as np
import sys
# Custom imports
from utils.cascade import getModel

if len(sys.argv) != 2:
    print "Usage : testExtended.py <model_file>"
    sys.exit(2)

# Define the model
model_file = sys.argv[1]
fprop_func = getModel(model_file)
print fprop_func

mini_batch = np.random.uniform(0, 256, size=(64, 27, 27, 3))
mini_batch = np.asarray(mini_batch, dtype=np.float32)
mini_batch = np.transpose(mini_batch, (3, 1, 2, 0))

print mini_batch.shape
results = fprop_func(mini_batch)
print results[0]
print results.shape

