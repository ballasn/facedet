#! /Tmp/lisa/os_v3/canopy/bin/python

import sys
import theano.tensor as T
from theano import function
from time import time
import cPickle as pkl
from os.path import exists
import imp
import os

# Custom imports
from facedet.utils.cascade.process import process_pascal


"""
This file will launch the detection on the images of AFW.
AFW consists of 205 images, but the resolution can go up to ~4000*4000
Write your parameters at <params.py> and use this file as argument for the
script.

Necessary variables in <params.py> :
====================================
models : list
    list of model files
sizes : list
    list of the input size of the models
strides : list
    list of the prediction stride of the models
cascade_scales : list of lists
    list of the list of scales to be used on the images
    one list per model
probs : list
    list of the thresholds, one per model
overlap_ratio : list
    list of the overlap_ratio allowed, one per model
min_pred_size : int
    size of the minimum window to be test
    this will be used to check scales, and remove the useless ones
piece_size : int
    size of the pieces to be cut from the whole image
    in case the whole image is too big prevents MemoryError on GPUs,
    and dimension problem with cudNN
max_img_size : int
    If the initial image is larger than the value,
    it will be resized with its max side becoming max_img_size.
    The reduction ratio will be used for the other side,
    so that the image is not deformed.
"""


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "usage %s : <params.py>" % sys.argv[0]
        sys.exit(2)

    # Load parameters
    params = imp.load_source('params', sys.argv[1])
    exp_name, ext = os.path.splitext(os.path.basename(sys.argv[1]))
    print params

    # Check inputs
    assert len(params.models) == len(params.sizes)
    assert len(params.models) == len(params.strides)
    assert len(params.models) == len(params.cascade_scales)
    assert len(params.models) == len(params.probs)
    assert len(params.models) == len(params.overlap_ratio)

    # Load models
    models_ = []
    for m in params.models:
        with open(m, 'r') as mf:
            mo_ = pkl.load(mf)
            mo_.layers.pop()
            models_.append(mo_)

    # Compile functions
    fprops = []
    x = T.tensor4('x')
    for m_ in models_:
        pred_ = function([x], m_.fprop(x))
        fprops.append(pred_)

    print 'Only patches with sizes > '+str(params.min_pred_size),
    print ' pixels should be tested'
    for i in range(len(params.cascade_scales)):
        params.cascade_scales[i] = [e for e in params.cascade_scales[i]
                                    if params.sizes[i]/e >= params.min_pred_size]
    print 'Scales used'
    for e in params.cascade_scales:
        print e

    # Dir at which it will write results
    out_dir = './results/'
    print out_dir
    assert exists(out_dir)

    # Detect faces
    t_orig = time()
    process_pascal(models_, fprops,
                   params.cascade_scales, params.sizes,
                   params.strides, params.probs,
                   params.overlap_ratio, out_dir,
                   min_pred_size=params.min_pred_size,
                   piece_size=params.piece_size,
                   max_img_size=params.max_img_size,
                   name=exp_name)
    t = time()
    print t-t_orig, 'seconds for AFW'
