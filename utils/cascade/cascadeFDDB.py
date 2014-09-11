import numpy as np
import sys
from os.path import join, split, isdir, isfile
from os import remove, mkdir, listdir
from optparse import OptionParser
import cPickle as pkl
import theano.tensor as T
from theano import function
from time import time

# Custom imports
from utils.cascade.protocol_test import cascade


def process_fold(fprops, scales, sizes, strides, probs,
                 nb_fold, out_dir,
                 fold_dir="/data/lisa/data/faces/FDDB/FDDB-folds/",
                 img_dir="/data/lisa/data/faces/FDDB/"):
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

    # Checking existing files
    L = []
    for f in files:
        if isfile(f):
            L.append(f)

    # Perform detection

    rois_tot = []
    scores_tot = []
    l_f = len(files)

    for i, f in enumerate(L):
        sys.stdout.write("\r" + str(nb_fold) + "th fold,"
                         + str(i) + "/" + str(l_f) + " processed images")
        sys.stdout.flush()

        # Perform cascade classification on image f
        rois, scores = cascade(f, fprops, scales, sizes, strides, probs)
        rois_tot.append(rois)
        scores_tot.append(scores)

    # Writing the results now

    output_fold = join(out_dir, "fold-"+nb_s+"-out.txt")
    with open(output_fold, 'wb') as output:
        for i, f in enumerate(L):
            # We need to format names to fit FDDB test
            # We remove /data/lisa/data/faces/FDDB and the extension
            n = f.split("/")[6:]
            n = "/".join(n)[:-4]

            output.write(f+"\n")  # Filename for FDDB
            output.write(str(len(rois_tot[i]))+"\n")  # Nb of faces

            # Faces for the image
            for roi, score in zip(rois_tot[i], scores_tot[i]):
                # <x0, y0, w, h, score>
                s = str(roi[0, 0]) + ',' + str(roi[0, 1])+','
                w = str(roi[1, 0] - roi[0, 0])
                h = str(roi[1, 1] - roi[0, 1])
                s += w + ',' + h + ',' + str(score)
                output.write(s + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "usage %s : <model16.pkl> <model48.pkl> <model96.pkl>"\
              % sys.argv[0]
        sys.exit(2)

    model_file16 = sys.argv[1]
    model_file48 = sys.argv[2]
    model_file96 = sys.argv[3]
    sizes = [16, 48, 96]
    strides = [1, 1, 1]
    scales = [0.5]
    probs = [0.2, 0.2, 0.2]
    nb_fold = 1
    out_dir = './output/'

    with open(model_file16, 'r') as m_f:
        model16 = pkl.load(m_f)

    with open(model_file48, 'r') as m_f:
        model48 = pkl.load(m_f)

    with open(model_file96, 'r') as m_f:
        model96 = pkl.load(m_f)

    # Compile functions
    x = T.tensor4('x')
    predict16 = function([x], model16.fprop(x))
    predict48 = function([x], model48.fprop(x))
    predict96 = function([x], model96.fprop(x))

    fprops =[predict16, predict48, predict96]
    t_orig = time()
    for nb in range(1,11):
        t0 = time()
        process_fold(fprops, scales, sizes, strides, probs,
                     nb, out_dir)
        t = time()
        print ""
        print t-t0, 'seconds for the fold'
    print t-t_orig, 'seconds for FDDB'

