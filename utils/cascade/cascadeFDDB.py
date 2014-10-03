import numpy as np
import sys
from os.path import join, split, isdir, isfile
from os import remove, mkdir, listdir
from optparse import OptionParser
import cPickle as pkl
import theano.tensor as T
from theano import function
from time import time
import cv2
from math import sqrt
# Custom imports
from utils.cascade.protocol_test import cascade


def process_fold(models, fprops, scales, sizes, strides, probs, overlap_ratio,
                 nb_fold, out_dir,
                 fold_dir2='/u/chassang/Projects/FaceDetection/FDDB_files_lists',
                 fold_dir="/data/lisa/data/faces/FDDB/FDDB-folds/",
                 img_dir="/data/lisa/data/faces/FDDB/",
                 mode='rect'):
    """
    Apply the cascade of fprops to the folds
    Write the results at <out_dir>/fold-<nb_fold>-out.txt
    """

    # define file indicating list of files
    if nb_fold < 10:
        nb_s = "0" + str(nb_fold)
    else:
        nb_s = str(nb_fold)

    # Define fold
    fold = join(fold_dir, "FDDB-fold-"+nb_s+".txt")
    print "Working on", fold
    # define list of files as a pyList
    files = []
    with open(fold, "rb") as fold_list:
        for line in fold_list:
            files.append(join(img_dir, line[:-1]+".jpg"))  # Remove \n


    # Perform detection

    rois_tot = []
    scores_tot = []
    l_f = len(files)
    for i, f in enumerate(files):

        sys.stdout.write("\r" + str(nb_fold) + "th fold,"
                         + str(i) + "/" + str(l_f) + " processed images | " + f)
        sys.stdout.flush()

        # Perform cascade classificiation on image f
        if isfile(f):
            img_ = cv2.imread(f)
            rois, scores = cascade(img_, models, fprops, scales, sizes, strides, overlap_ratio, probs)
            rois_tot.append(rois)
            scores_tot.append(scores)
        else:
            rois_tot.append([])
            scores_tot.append([])
    # Writing the results now

    output_fold = join(out_dir, "fold-"+nb_s+"-out.txt")
    with open(output_fold, 'wb') as output:
        for i, f in enumerate(files):
            # We need to format names to fit FDDB test
            # We remove /data/lisa/data/faces/FDDB and the extension
            n = f.split("/")[6:]
            n = "/".join(n)[:-4]

            output.write(n+"\n")  # Filename for FDDB
            output.write(str(len(rois_tot[i]))+"\n")  # Nb of faces

            # Faces for the image
            for roi, score in zip(rois_tot[i], scores_tot[i]):

                # <x0, y0, w, h, score>
                s = str(roi[0, 1]) + ' ' + str(roi[0, 0])+' '
                w = roi[1, 1] - roi[0, 1]
                h = roi[1, 0] - roi[0, 0]
                if mode == 'rect':
                    s += str(w) + ' ' + str(h) + ' '
                elif mode == 'ellipse':
                    s = str(w/2) + ' ' + str(h/2) + ' 0 ' + s
                output.write(s + str(score) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "usage %s : <model16.pkl> <model48.pkl> <model96.pkl>"\
              % sys.argv[0]
        sys.exit(2)

    model_file16 = sys.argv[1]
    model_file48 = sys.argv[2]
    model_file96 = sys.argv[3]


    #for i in xrange(len(sizes)):
    #    local_scales.append([s * float(sizes[i]) / base_size
    #                         for s in global_scales])


    nb_fold = 1
    out_dir = './results/output/'

    with open(model_file16, 'r') as m_f:
        model16 = pkl.load(m_f)

    with open(model_file48, 'r') as m_f:
        model48 = pkl.load(m_f)

    #with open(model_file96, 'r') as m_f:
    #    model96 = pkl.load(m_f)

    # Compile functions
    x = T.tensor4('x')
    predict16 = function([x], model16.fprop(x))
    predict48 = function([x], model48.fprop(x))
    #predict96 = function([x], model96.fprop(x))

    models = [model16]
    fprops = [predict16]
    sizes = [16]
    strides = [2]
    base_size = max(sizes)
    probs = [0.5]
    overlap_ratio = [0.6]


    ratio = sqrt(2)
    global_scales = [0.05 * ratio**e for e in range(5)]
    local_scales = [global_scales]
    print 'local_scales', local_scales


    assert len(models) == len(fprops)
    assert len(models) == len(sizes)
    assert len(models) == len(strides)
    assert len(models) == len(local_scales)


    t_orig = time()
    for nb in range(1, 11):
        t0 = time()
        process_fold(models, fprops, local_scales, sizes, strides, probs, overlap_ratio,
                     nb, out_dir, mode='rect')
        t = time()
        print ""
        print t-t0, 'seconds for the fold'
    print t-t_orig, 'seconds for FDDB'
