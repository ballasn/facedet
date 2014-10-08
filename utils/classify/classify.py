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

    if len(sys.argv) < 3:
        print("Usage %s: model outputdir [nb_run_model]" % sys.argv[0])
        exit(1)



    outputdir =  sys.argv[2]
    nb_run_model = 1
    if len(sys.argv) >= 4:
        nb_run_model = int(sys.argv[3])

    ### Load model
    model = None
    with open(sys.argv[1], "rb") as file:
        model = cPickle.load(file)
    print(model)

    ### Load dataset
    dataset = faceDataset(
        which_set = 'valid',
        ratio = 0.9,
        positive_samples="/data/lisatmp3/ballasn/facedet/datasets/aflw/feats16.npy",
        #negative_samples="/data/lisatmp3/ballasn/facedet/datasets/pascal/neg100000_16.npy",
        negative_samples="/data/lisatmp3/ballasn/facedet/datasets/googleemotion/neg100_good_shuffled.npy",
        #mean='/data/lisatmp3/ballasn/facedet/datasets/aflw/mean_16pascal.npy')
        mean='/data/lisatmp3/ballasn/facedet/datasets/aflw/mean_16google.npy')

    ### Compile network prediction function
    tensor5 = T.TensorType(config.floatX, (False,)*5)
    x = T.tensor4("x")
    predict = function([x], model.fprop(x))

    ### Compute prediction of training set
    batch_size = 128
    nb_classes = 2
    print dataset.nb_examples
    preds  = np.zeros((dataset.nb_examples, nb_classes), dtype=config.floatX)
    ## Binary labels
    blabels = np.zeros((dataset.nb_examples, nb_classes), dtype=config.floatX)


    for r in xrange(0, nb_run_model):
        print "run model", r
        cur = 0
        batches = dataset.iterator(batch_size=batch_size)
        for x, y in batches:
            #print(cur, np.argmax(y, axis=1))
            pred_tmp = np.zeros((batch_size, nb_classes), dtype=config.floatX)
            pred_tmp = predict(x)

            #print pred_tmp.shape
            pred_tmp = np.reshape(pred_tmp, (128, 2))


            preds[cur*batch_size:(cur+1)*batch_size] += pred_tmp
            blabels[cur*batch_size:(cur+1)*batch_size] = y
            cur = cur +1

    preds /= nb_run_model

    print(blabels.shape)
    print(preds.shape)

    #np.save(os.path.join(outputdir, 'pred.npy'), preds)
    #np.save(os.path.join(outputdir, 'labels.npy'), blabels)

    ### Keep only the original number of example
    ### (Not necessary a multiple of batch_size)
    nb_examples = dataset.nb_examples
    #nb_examples = dataset.orig_nb_examples
    #nb_examples = dataset.orig_nb_examples
    #preds = preds[0:nb_examples, :]
    #blabels = blabels[0:nb_examples, :]

    ### Transform pred in outofc binary vector
    labels = oneofc_inv(blabels)
    prlabels = oneofc_inv(preds)
    #reflabels = oneofc_inv(dataset.labels[0:nb_examples, :])
    #assert np.equal(labels, reflabels).all()

    ### Create list to save good_classif and missclassif
    goodclassif = []
    missclassif = []
    missclasslabels = []
    for i in xrange(0, nb_classes):
        goodclassif.append([])
        missclassif.append([])
        missclasslabels.append([])


    ### Compute Score
    # Print scores and save missclassified files
    for i in xrange(0, preds.shape[0]):
        #print i, preds[i, :], prlabels[i], labels[i]
        if prlabels[i] == labels[i]:
            goodclassif[labels[i]].append(i)
        else:
            missclassif[labels[i]].append(i)
            missclasslabels[labels[i]].append(prlabels[i])


    pre = np.zeros(nb_classes)
    rec = np.zeros(nb_classes)
    ### Compute true/false positive, negative
    for c in xrange(0, nb_classes):
        tp = np.sum((labels == c) & (prlabels == c))
        fp = np.sum((labels != c) & (prlabels == c));
        tn = np.sum((labels != c) & (prlabels != c));
        fn = np.sum((labels == c) & (prlabels != c));

        pre[c] = tp / (tp + fp + 0.0001)
        rec[c] = tp / (tp + fn + 0.0001)
        print c, pre[c], rec[c]

    print "Acc (avg):", np.mean(pre, axis=0), "Rec (avg):", np.mean(rec, axis=0)

    ### Write goodclassif and missclassif
    for i in xrange(0, nb_classes):
        label_name = str(i)
        filename = os.path.join(outputdir, "goodclassif_" + label_name + ".txt")
        with open(filename, "w") as fd:
            for item in goodclassif[i]:
                fd.write("%d\n" % item)
        filename = os.path.join(outputdir, "missclassif_" + label_name + ".txt")
        with open(filename, "w") as fd:
            for j in xrange(0, len(missclassif[i])):
                fd.write("%d %d\n" % (missclassif[i][j],
                                      missclasslabels[i][j]))

