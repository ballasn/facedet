#! /Tmp/lisa/os_v3/canopy/bin/python
"""
Takes a file and apply the given regression to the bounding boxes.
This assumes that the results are given as :
img_file score x0 y0 x1 y1
"""

import sys
import theano.tensor as T
from theano import function
from time import time
import numpy as np
import cv2
import cPickle as pkl
# Custom imports


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print "usage %s : <model.pkl> <input_file> <regressor.pkl> <output_file>"\
              % sys.argv[0]
        sys.exit(2)
    # Image directory
    img_dir = "/data/lisa/data/faces/AFW/testimages/"
    patch_size = 96

    model_file = sys.argv[1]
    input_file = sys.argv[2]
    regressor_file = sys.argv[3]
    output_file = sys.argv[4]

    with open(model_file, 'r') as m_f:
        model = pkl.load(m_f)
    with open(regressor_file, 'r') as m_f:
        regressor = pkl.load(m_f)

    # Compile functions
    # Remove sigmoid
    model.layers.pop()
    # Remove fully connected
    model.layers.pop()
    x = T.tensor4('x')
    detect = function([x], model.fprop(x))
    print 'Compiled detector'
    x = T.matrix('x')
    regress = function([x], regressor.fprop(x),allow_input_downcast=True)
    print 'Compiled regressor'

    with open(input_file, 'r') as inp:
        lines = inp.read().splitlines()
    s = ''
    entry = np.zeros((1, 8194))
    for line in lines:
        row = line.split(' ')
        # The file is not exactly what is indicated
        #  we need to add the base directory in front of it
        img = cv2.imread(img_dir+row[0]+'.jpg')
        [x0, y0, x1, y1] = [int(e) for e in row[2:]]
        patch = img[max(0, int(y0)):
                    min(img.shape[0], int(y1)),
                    max(0, int(x0)):
                    min(img.shape[1], int(x1)), :]
        patch = cv2.resize(patch, (patch_size, patch_size),
                           interpolation=cv2.INTER_CUBIC)
        patch = np.reshape(patch, [1]+list(patch.shape))
        patch = np.transpose(patch, (0, 3, 1, 2))
        feats = detect(patch)
        # We don't care about the score
        entry[0, 0:2] = [x1 - x0, y1 - y0]
        feats = np.asarray(feats, dtype='float32')
        entry[0, 2:] = np.copy(np.reshape(feats,
                    feats.shape[1]*feats.shape[2]*feats.shape[3]))
        # The output only consists in [x, y, w, h]
        output = regress(entry)
        output = output[0,:]
        output[2] = max(output[2], 10)
        output[3] = max(output[3], 10)
        # Writing the new results : id score x0 y0 x1 y1
        new_line = row[0]+' '+row[1]+' '
        new_line += str(max(int(output[0]+x0), 0))+' '
        new_line += str(max(int(output[1]+y0), 0))+' '
        new_line += str(max(int(output[0]+output[2]+x0), 0))+' '
        new_line += str(max(int(output[1]+output[3]+y0), 0))+'\n'

        s += new_line

    with open(output_file, 'w') as out:
        out.write(s)

    print 'Done regressing the data !'
