import sys
import os
import cPickle
import gzip
from itertools import islice

from theano import tensor as T
from theano import config
from theano import function
import numpy as np


from facedet.datasets.faceDataset import faceDataset


def oneofc(pred, nb_classes):
    opred = zeros((pred.shape[0], nb_classes))
    for i in  pred.shape[0]:
        opred[i, pred[i]] =1
    return opred

def oneofc_inv(pred):
    out = np.argmax(pred, axis=1)
    return out



if __name__ == '__main__':

    if len(sys.argv) < 5:
        print("Usage %s: model positive negative relax_value" % sys.argv[0])
        exit(1)


    relax = int(sys.argv[4])


    ### Load model
    model = None
    with open(sys.argv[1], "rb") as file:
        model = cPickle.load(file)
    print(model)

    ### Load dataset
    dataset = faceDataset(
        which_set = 'valid',
        ratio = 0.9,
        positive_samples=sys.argv[2],
        negative_samples=sys.argv[3])


    ### Perform relaxation
    tparams = model.layers[-1].get_param_values()
    tparams = [x/float(self.relaxation_term) for x in tparams]
    model.layers[-1].set_param_values(tparams)

    ### Compile network prediction function
    x = T.tensor4("x")
    predict = function([x], model.fprop(x))

    ### Compute prediction of training set
    batch_size = 128
    nb_classes = 2
    print dataset.nb_examples
    preds  = np.zeros((dataset.nb_examples, nb_classes), dtype=config.floatX)
    ## Binary labels
    blabels = np.zeros((dataset.nb_examples, nb_classes), dtype=config.floatX)


    cur = 0
    batches = dataset.iterator(batch_size=batch_size, data_specs=['x', ['x', 'y']])
    for x, y in batches:
        #print(cur, np.argmax(y, axis=1))
        pred_tmp = np.zeros((batch_size, nb_classes), dtype=config.floatX)
        pred_tmp = predict(x)

        print pred_tmp


        #print pred_tmp.shape
        pred_tmp = np.reshape(pred_tmp, (128, 2))
        preds[cur*batch_size:(cur+1)*batch_size] += pred_tmp
        blabels[cur*batch_size:(cur+1)*batch_size] = y
        cur = cur +1

    print(blabels.shape)
    print(preds.shape)

