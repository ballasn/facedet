#! /Tmp/lisa/os_v3/canopy/bin/python
"""
Takes a file and apply the given regression to the bounding boxes.
This assumes that the results are given as :
<img_file>
<nb_boxes>
x1 y1 w1 h1 score1
...
x_nb y_nb...
"""

import sys
import theano.tensor as T
from theano import function
import cPickle as pkl
from os.path import join
# Custom imports
import facedet.utils.cascade.reg_utils as reg_utils


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print "usage %s : <model.pkl> <input_file> <regressor.pkl> <img_list> <output_file>"\
              % sys.argv[0]
        sys.exit(2)
    # Image directory
    img_dir = "/data/lisa/data/faces/FDDB/"
    patch_size = 96

    model_file = sys.argv[1]
    input_dir = sys.argv[2]
    regressor_file = sys.argv[3]
    img_list_dir = sys.argv[4]
    output_dir = sys.argv[5]

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
    regress = function([x], regressor.fprop(x), allow_input_downcast=True)
    print 'Compiled regressor'

    # Read the list of img ids
    for nb in range(1, 11):
        if nb < 10:
            nb_st = '0'+str(nb)
        else:
            nb_st = str(nb)
        # define files to be read and written
        l_f = 'FDDB-folds-'+nb_st+'.txt'
        i_f = 'fold-'+nb_st+'-out.txt'
        img_list_file = join(img_list_dir, l_f)
        input_file = join(input_dir, i_f)
        output_file = join(input_dir, i_f)
        print 'reading from', input_file,
        print 'writing at', output_file,

        with open(img_list_file, 'r') as ifl:
            img_list = ifl.read().splitlines()

        # Read the detections to a dict
        detections = reg_utils.read_to_dict(input_file, data='rect', wh=True)

        reg_detections = reg_utils.regress_dict(detections, detect, regress,
                                                img_dir,
                                                patch_size=96,
                                                entry_size=512*4*4+2)
        st = reg_utils.dictToFddbStyle(reg_detections, img_list)
        with open(output_file, 'w')as of:
            of.write(st)
    print 'Done regressing and writing the data !'
