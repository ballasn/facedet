#! /Tmp/lisa/os_v3/canopy/bin/python

import sys
import theano.tensor as T
from theano import function
from time import time
import cPickle as pkl
from os.path import exists
import importlib
# Custom imports
from utils.cascade.AFW.processAFW import process_pascal


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print "usage %s : <model16.pkl> <model48.pkl> <model96.pkl> <params.py>"\
              % sys.argv[0]
        sys.exit(2)

    model_file16 = sys.argv[1]
    model_file48 = sys.argv[2]
    model_file96 = sys.argv[3]
    # Params :
    exp_name = sys.argv[4][:-3]
    params = importlib.import_module(exp_name)

    with open(model_file48, 'r') as m_f:
        model48 = pkl.load(m_f)

    # Compile functions
    model48.layers.pop()
    x = T.tensor4('x')
    predict48 = function([x], model48.fprop(x))

    models = [model48]
    fprops = [predict48]
    sizes = [48]
    strides = [8]
    base_size = max(sizes)

    # Check inputs
    assert len(models) == len(fprops)
    assert len(models) == len(sizes)
    assert len(models) == len(strides)
    assert len(models) == len(params.local_scales)

    print 'Only patches with sizes > '+str(params.min_pred_size),
    print ' pixels should be tested'
    for i in range(len(params.local_scales)):
        params.local_scales[i] = [e for e in params.local_scales[i]
                                  if sizes[i]/e >= params.min_pred_size]
    print 'Scales used'
    for e in params.local_scales:
        print e

    # AFW dataset : 205 images (these are large ones)
    out_dir = '../../AFW/face-eval/detections/AFW/'
    print out_dir
    assert exists(out_dir)
    t_orig = time()
    process_pascal(models, fprops, params.local_scales, sizes, strides, params.probs,
                   params.overlap_ratio, out_dir,
                   min_pred_size=params.min_pred_size,
                   piece_size=params.piece_size,
                   max_img_size=params.max_img_size,
                   name='')
    t = time()
    print t-t_orig, 'seconds for AFW'
