import sys
import cv2
from os.path import join
import numpy as np

import cPickle as pkl
import theano.tensor as T
from theano import function

base_dir = '/data/lisa/data/faces/FDDB/'
feats_base='/u/ballasn/project/facedet/facedet/utils/FDDB/bbox2ellipse/regression/data/feats_fold'
labels_base='/u/ballasn/project/facedet/facedet/utils/FDDB/bbox2ellipse/regression/data/labels_fold'


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print "usage %s : <feats> <labels> <fold_nb>" % sys.argv[0]
        sys.exit(2)


    fold_nb = int(sys.argv[3])

    feats_dim = 0
    feats_nb = 0
    labels_dim = 0
    for i in xrange(1, 11):
        feats_file = feats_base + str(i) + ".npy"
        feats = np.load(feats_file)
        labels_file = labels_base + str(i) + ".npy"
        labels = np.load(labels_file)


        labels_dim = labels.shape[1]
        feats_dim = feats.shape[1]
        feats_nb += feats.shape[0]


    print feats_nb, feats_dim, labels_dim
    feats = np.zeros((feats_nb, feats_dim))
    labels = np.zeros((feats_nb, labels_dim))

    cur = 0
    for i in xrange(1, 11):
        if i == fold_nb:
            continue
        feats_file = feats_base + str(i) + ".npy"
        curfeats = np.load(feats_file)
        labels_file = labels_base + str(i) + ".npy"
        curlabels = np.load(labels_file)


        feats[cur:cur+curfeats.shape[0], :] = curfeats[:, :]
        labels[cur:cur+curfeats.shape[0], :] = curlabels[:, :]
        cur += curfeats.shape[0]

    np.save(sys.argv[1], feats)
    np.save(sys.argv[2], labels)




