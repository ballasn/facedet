import sys
import os
import cPickle
import gzip
from itertools import islice

from theano import tensor as T
from theano import config
from theano import function
import numpy as np


from facedet.datasets.faceDataset_cascade import faceDataset

def oneofc(pred, nb_classes):
    opred = zeros((pred.shape[0], nb_classes))
    for i in  pred.shape[0]:
        opred[i, pred[i]] =1
    return opred

def oneofc_inv(pred):
    out = np.argmax(pred, axis=1)
    return out

def inc_idx(val, lst):
    while val in lst:
        val = val + 1


if __name__ == '__main__':

    positive_samples = "FIXME"
    negative_samples = "FIXME"
    ratio = 0.8

    if len(sys.argv) < 4:
        print("Usage %s: model prob inactive_example_out [inactive_example_in]" % sys.argv[0])
        exit(1)


    acc_prob = float(sys.argv[2])
    out =  sys.argv[3]
    if len(sys.argv) >= 5:
        inactive = sys.argv[4]
    else:
        inactive = None

    ### Load model
    model = None
    with open(sys.argv[1], "rb") as file:
        model = cPickle.load(file)
    print(model)

    ### Load dataset
    ### Note that we test on all example each time, this could be optimized
    dataset = faceDataset(positive_samples, negative_samples, 'test',
                          ratio)

    ### Compile network prediction function
    x = T.tensor4("x")
    predict = function([x], model.fprop(x))

    ### Compute prediction of training set
    batch_size = 128
    nb_classes = 2
    preds  = np.zeros((dataset.nb_examples, nb_classes), dtype=config.floatX)


    ###
    if inactive != None:
        inactive = np.load(inactive)
    else:
        inactive = []

    cur = 0
    batches = dataset.iterator(mode='negative_seq',
                               batch_size=batch_size)
    pred = np.zeros((batch_size, nb_classes), dtype=config.floatX)
    for x, y in batches:
        pred = predict(x)
        prlabels = oneofc_inv(pred)

        for i in shape(pred, 0):
            if y[i, 1] == 1 and prlabels[i, 0] == 0:
                if pred[i, 1] < acc_prob:
                    inactive.append(cur)
            cur += 1


    inactive = np.unique(inactive)
    np.save(sys.argv[3], inactive)
