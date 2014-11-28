from theano import function
import theano.tensor as T
import sys
import cPickle as pkl
from time import time
from os.path import exists, splitext, basename
import imp
from facedet.utils.cascade.processFDDB import process_fold


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage %s : <params.py>"\
              % sys.argv[0]
        sys.exit(2)

    # Load parameters
    params = imp.load_source('params', sys.argv[1])
    exp_name, ext = splitext(basename(sys.argv[1]))
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
        params.cascade_scales[i] = [e for e in params.cascade_scales[i] if
                                    params.sizes[i]/e >= params.min_pred_size]
    print 'Scales used'
    for e in params.cascade_scales:
        print e

    # Dir at which it will write results
    out_dir = './results/'
    print 'Will write at', out_dir
    assert exists(out_dir)

    # Detect faces
    t_orig = time()
    fake_fold = '../cascade/fake_fold/'
    for nb in range(10, 11):
        t0 = time()
        process_fold(models_, fprops,
                     params.cascade_scales, params.sizes, params.strides,
                     params.probs, params.overlap_ratio,
                     nb,
                     out_dir,
                     #fold_dir=fake_fold,
                     mode='rect')
        t = time()
        print ""
        print t-t0, 'seconds for the fold'
    print t-t_orig, 'seconds for FDDB'
