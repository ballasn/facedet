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
                 mode='rect',
                 max_det=60):
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
            m = min(max_det, len(rois_tot[i]))

            output.write(n+"\n")  # Filename for FDDB
            output.write(str(m)+'\n')  # len(rois_tot[i]))+"\n")  # Nb of faces
            # Faces for the image
            for roi, score in zip(rois_tot[i][:m], scores_tot[i][:m]):

                # <x0, y0, w, h, score>
                s = str(roi[0, 1]) + ' ' + str(roi[0, 0])+' '
                w = roi[1, 1] - roi[0, 1]
                h = roi[1, 0] - roi[0, 0]
                if mode == 'rect':
                    s += str(w) + ' ' + str(h) + ' '
                elif mode == 'ellipse':
                    s = str(w/2) + ' ' + str(h/2) + ' 0 ' + s
                output.write(s + str(score) + '\n')


def process_pascal(models, fprops, scales, sizes, strides, probs, overlap_ratio,
                   out_dir,
                   id_list="/data/lisa/data/faces/AFW/annot/list",
                   img_dir="/data/lisa/data/faces/AFW/testimages/",
                   max_det=30):
    """
    Apply the cascade of fprops to the folds
    Write the results at <out_dir>/res-out.txt in pascal VOC format
    """

    # Read imgs list
    files = []
    ids = []
    with open(id_list) as _f:
        for line in _f:
            ids.append(line[:-1])  # Remove \n
            files.append(join(img_dir, line[:-1]+".jpg"))  # Remove \n


    # Perform detection
    rois_tot = []
    scores_tot = []
    l_f = len(files)
    for i, f in enumerate(files):
        sys.stdout.write(str(i) + "/" + str(l_f) + " processed images | " + f)
        sys.stdout.flush()
        # Perform cascade classificiation on image f
        if isfile(f):
            img_ = cv2.imread(f)
            ### If max dim > 640, resize
            if max(img_.shape[0], img_.shape[1]) > 240:
                if img_.shape[0] > img_.shape[1]:
                    rs = 240.0 / img_.shape[0]
                else:
                    rs = 240.0 / img_.shape[1]
                print rs, int(img_.shape[0] * rs), int(img_.shape[1] * rs)
                #exit(1)
                img_ = cv2.resize(img_, (int(img_.shape[1] * rs), int(img_.shape[0] * rs)))
                #cv2.imshow("input", img_)
                #cv2.waitKey()

            rois, scores = cascade(img_, models, fprops, scales, sizes, strides, overlap_ratio, probs)
            rois_tot.append(rois)
            scores_tot.append(scores)
            print " :", len(rois)
        else:
            rois_tot.append([])
            scores_tot.append([])

    # Writing detections in PascalVOC format
    outputf = join(out_dir, "detection.txt")
    with open(outputf, 'w') as output:
        for i, f in enumerate(files):
            if max_det != -1:
                m = min(max_det, len(rois_tot[i]))
            else:
                m = len(rois_tot[i])

            #output.write(+"\n")  # Filename for FDDB
            #output.write(str(m)+'\n')  # len(rois_tot[i]))+"\n")  # Nb of faces
            # Faces for the image
            for roi, score in zip(rois_tot[i][:m], scores_tot[i][:m]):

                # <img_id score left top right bottom>
                x = int(np.floor(roi[0, 1] * (1-rs)))
                y = int(np.floor(roi[0, 0] * (1-rs)))
                w = int(np.floor(roi[1, 1] - roi[0, 1]) * (1-rs))
                h = int(np.floor(roi[1, 0] - roi[0, 0]) * (1-rs))

                output.write(ids[i] + ' ' + str(score)+ ' ')
                output.write(str(y) + ' ' + str(x) + ' ' + str(y+h) + ' ' +
                        str(x+w) + '\n')
                print ids[i] + ' ' + str(score), str(y) + ' ' + str(x) + ' ' +\
                        str(y+h) + ' ' + str(x+w)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "usage %s : <model16.pkl> <model48.pkl> <model96.pkl>"\
              % sys.argv[0]
        sys.exit(2)


    mode = "fddb"
    #mode = "pascal"

    model_file16 = sys.argv[1]
    model_file48 = sys.argv[2]
    model_file96 = sys.argv[3]


    #for i in xrange(len(sizes)):
    #    local_scales.append([s * float(sizes[i]) / base_size
    #                         for s in global_scales])


    nb_fold = 1
    out_dir = 'results/output96/'

    #with open(model_file16, 'r') as m_f:
    #    model16 = pkl.load(m_f)

    with open(model_file48, 'r') as m_f:
        model48 = pkl.load(m_f)
    model48.layers.pop()


    with open(model_file96, 'r') as m_f:
        model96 = pkl.load(m_f)
    model96.layers.pop()

    # Compile functions
    x = T.tensor4('x')
    #predict16 = function([x], model16.fprop(x))
    predict48 = function([x], model48.fprop(x))
    predict96 = function([x], model96.fprop(x))


    #models = [model16, model48]
    #fprops = [predict16, predict48]
    #sizes = [48 , 48]
    #strides = [1, 1]
    #base_size = max(sizes)
    #probs = [0 , 2]
    #overlap_ratio = [0.5 , 0.3]
    #
    #ratio = sqrt(2)
    #global_scales = [(1.0/ratio)**e for e in range(0, 8)]
    #global_scales2 = [(1.0/ratio)**e for e in range(0, 8)]
    #local_scales = [global_scales, global_scales2]
    # print 'local_scales', local_scales


    models = [model48]
    fprops = [predict48]
    sizes = [48]
    strides = [1]
    base_size = max(sizes)
    probs = [-1]
    overlap_ratio = [0.3]

    models = [model96]
    fprops = [predict96]
    sizes = [96]
    strides = [1]
    base_size = max(sizes)
    probs = [-1]
    overlap_ratio = [0.3]


    models = [model48, model96]
    fprops = [predict48, predict96]
    sizes = [48, 96]
    strides = [1, 1]
    base_size = max(sizes)
    probs = [-1, 0]
    overlap_ratio = [0.5, 0.3]


    ratio = sqrt(2)
    global_scales = [(1.0/ratio)**e for e in range(0, 11)]
    global_scales2 = [(1.0/ratio)**e for e in range(0, 11)]
    local_scales = [global_scales, global_scales2]
    #local_scales = [global_scales2]
    local_scales[0].append(1.2)
    local_scales[0].append(1.4)
    local_scales[-1].append(1.2)
    local_scales[-1].append(1.4)
    local_scales[-1].append(1.6)
    #local_scales[-1].append(1.8)
    #local_scales[-1].append(2.0)
    #local_scales[-1].append(2.2)
    #local_scales[-1].append(2.4)
    #local_scales[-1].append(2.8)
    print 'local_scales', local_scales

    # Check that the smallest patch is larger than 20 px
    # patch_size = predict_size / scale
    print 'Only patches with sizes > 20 pixels should be tested'
    for i in range(len(local_scales)):
        local_scales[i] = [e for e in local_scales[i] if sizes[i]/e >= 20]
    print 'local_scales', local_scales

    assert len(models) == len(fprops)
    assert len(models) == len(sizes)
    assert len(models) == len(strides)
    assert len(models) == len(local_scales)

    if mode == "fddb":
        t_orig = time()
        for nb in range(1, 11):
            t0 = time()
            process_fold(models, fprops, local_scales, sizes, strides, probs,
                         overlap_ratio, nb, out_dir, mode='rect')
            t = time()
            print ""
            print t-t0, 'seconds for the fold'
        print t-t_orig, 'seconds for FDDB'
    else:
        process_pascal(models, fprops, local_scales, sizes, strides, probs,
                       overlap_ratio, out_dir)
