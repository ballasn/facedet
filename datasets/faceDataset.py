import os
import math
import cv2
import warnings
import numpy as np
import random
import thread
import time
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far DenseFeat is "
            "only supported with PyTables")

from theano import config
from pylearn2.datasets import dataset
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace





class faceDataset(dataset.Dataset):

    mapper = {'train': 0, 'valid': 1}

    def __init__(self,
                 positive_samples,
                 negative_samples,
                 which_set,
                 ratio = 0.8,
                 batch_size = 128,
                 axes = ('c', 0, 1, 'b')):
        """
        Instantiates a handle to the face dataset
        -----------------------------------------
        positive_samples : path to the npy file of + samples
        negative_samples : path to the npy file of - samples
        The current ratio is 0.8 => train 80%, valid 20%
        """

        if which_set == 'train':
            self.positives =  np.load(positive_samples)
            nb_train = int(np.ceil(ratio * self.positives.shape[0]))
            self.positives = self.positives[0:nb_train, :]
            print "positives train shape", self.positives.shape
            self.negatives = np.load(negative_samples)
            nb_train = int(np.ceil(ratio * self.negatives.shape[0]))
            self.negatives = self.negatives[0:nb_train, :]
            print "negatives train shape", self.negatives.shape

        elif which_set == 'valid':
            self.positives = np.load(positive_samples)
            nb_train = int(np.ceil(ratio * self.positives.shape[0]))
            self.positives = self.positives[nb_train:self.positives.shape[0], :]
            self.negatives = np.load(negative_samples)
            nb_train = int(np.ceil(ratio * self.negatives.shape[0]))
            self.negatives = self.negatives[nb_train:self.negatives.shape[0], :]
            print "positives",self.positives.shape
            print "negatives",self.negatives.shape
        ### duplicate last line to have nb_pos and nb_neg divisible by batch_size/2
        batch_size = batch_size / 2
        self.nb_pos = self.positives.shape[0]
        self.nb_neg = self.negatives.shape[0]

        if (self.nb_pos % batch_size != 0):
            to_add = batch_size - self.nb_pos % batch_size
            for k in range(0, to_add):
                self.positives = np.append(self.positives,
                                           self.positives[-1, :].reshape(1, self.positives.shape[1]),
                                           axis=0)
        if (self.nb_neg % batch_size != 0):
            to_add = batch_size - self.nb_neg % batch_size
            for k in range(0, to_add):
                self.negatives = np.append(self.negatives,
                                           self.negatives[-1, :].reshape(1, self.negatives.shape[1]),
                                           axis=0)

        self.img_shape = [48, 48, 3]
        self.nb_examples = self.positives.shape[0] + self.negatives.shape[0]
        self.which_set = which_set
        self.axes = axes


    def get_minibatch(self, cur_positives, cur_negatives,
                      minibatch_size,
                      data_specs, return_tuple):


        ### Initialize data
        x = np.zeros([minibatch_size, self.positives.shape[1]],
                     dtype = "float32")
        y = np.zeros([minibatch_size, 2],
                     dtype="float32")


        ### Get number of positives and negatives examples
        #nb_pos = int(0.5 * minibatch_size)
        nb_pos = int(np.random.rand() * minibatch_size)
        #print nb_pos,
        nb_neg = minibatch_size - nb_pos

        ### nb_examples must be divisible by minibatch_size

        if (cur_negatives + nb_neg >= self.negatives.shape[0]):
            nb_neg = self.negatives.shape[0] - cur_negatives
            nb_pos =  minibatch_size - nb_neg
        if (cur_positives + nb_pos >= self.positives.shape[0]):
            nb_pos = self.positives.shape[0] - cur_positives
            nb_neg =  minibatch_size - nb_pos

        ### Fill minibatch
        #print "cur_positives, nb_pos, cur_negatives,nb_neg"
        #print cur_positives, nb_pos, cur_negatives,nb_neg
        x[0:nb_pos, :] = self.positives[cur_positives:cur_positives+nb_pos, :]
        y[0:nb_pos, 0] = 1
        x[nb_pos:nb_pos+nb_neg, :] = self.negatives[cur_negatives:cur_negatives+nb_neg, :]
        y[nb_pos:nb_pos+nb_neg, 1] = 1

        x = np.reshape(x, [minibatch_size] + self.img_shape)
        #print x[0].shape
        #cv2.imshow("positif",np.asarray(x[0],dtype=np.uint8))
        #cv2.imshow("negatif",np.asarray(x[nb_pos],dtype=np.uint8))
        #cv2.waitKey(0)
        x = np.swapaxes(x, 0, 3)
        cur_positives += nb_pos
        cur_negatives += nb_neg
        ### Displaying pictures to check that pos!=neg

        return (x, y), cur_positives, cur_negatives



    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):

        # FIXME add  mode, topo and targets
        return FaceIterator(self, batch_size, num_batches,
                            data_specs, return_tuple, rng)

    def has_targets(self):
        return True

    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """
        return self.axes.index('b')

    def get_design_matrix(self, topo=None):
        return self.positives

    def get_num_examples(self):
        return self.nb_pos + self.nb_neg


def load_list(filename):
    id_list = []
    with open(filename, 'r') as fd:
        for line in fd:
            id_list.append(line)
    return id_list


class FaceIterator:

    def __init__(self, dataset=None, batch_size=None, num_batches=None,
                 data_specs=False, return_tuple=False, rng=None):

        self._dataset = dataset
        self._dataset_size = dataset.nb_examples

        # Validate the inputs
        assert dataset is not None
        if batch_size is None and num_batches is None:
            raise ValueError("Provide at least one of batch_size or num_batches")
        if batch_size is None:
            batch_size = int(np.ceil(self._dataset_size / float(num_batches)))
        if num_batches is None:
            num_batches = np.ceil(self._dataset_size / float(batch_size))

        max_num_batches = np.ceil(self._dataset_size / float(batch_size))
        if num_batches > max_num_batches:
            raise ValueError("dataset of %d examples can only provide "
                             "%d batches with batch_size %d, but %d "
                             "batches were requested" %
                             (self._dataset_size, max_num_batches,
                              batch_size, num_batches))

        if rng is None:
            self._rng = random.Random(1)
        else:
            self._rng = rng

        self._batch_size = batch_size
        self._num_batches = int(num_batches)
        self._cur_pos = 0
        self._cur_neg = 0
        self.stochastic = False

        self._num_pos = self._dataset.positives.shape[0]
        self._num_neg = self._dataset.negatives.shape[0]

        self._return_tuple = return_tuple
        self._data_specs = data_specs

        self.num_examples = self._dataset_size # Needed by Dataset interface
        print self.num_examples


    def __iter__(self):
        return self

    def next(self):
        if self._cur_pos >= self._num_pos and self._cur_neg >= self._num_neg:
            print "stopIteration :"
            print self.num_examples
            print self._num_batches
            raise StopIteration()
        else:
            data, self._cur_pos, self._cur_neg = \
                self._dataset.get_minibatch(self._cur_pos, self._cur_neg,
                                            self._batch_size, self._data_specs,
                                            self._return_tuple)
            return data



if __name__=="__main__":
    pos = "/data/lisatmp3/ballasn/facedet/positives.npy"
    neg = "/data/lisatmp3/ballasn/facedet/negatives.npy"
    print "instantiating datasets"
    fd_train = faceDataset(pos,neg,"train")
    fd_test = faceDataset(pos,neg,"valid")
    print "Done, now the iterator"
    print type(fd_train)
    print type(fd_test)
    fi_train = FaceIterator(dataset=fd_train, batch_size=128)
    fi_test = FaceIterator(dataset=fd_test, batch_size=128)
    b_train = fi_train.next()
    b_test = fi_test.next()
    print b_train[0].shape
    print b_train[1].shape
    x_train, y_train =  b_train[0], b_train[1]
    x_test, y_test = b_test[0], b_test[1]

    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape


    for i,(e,f) in enumerate(zip(y_train,y_test)):
            print e,f

    x_test = np.swapaxes(x_test,0,3)
    x_train = np.swapaxes(x_train,0,3)
    print x_test.shape
    print x_train.shape

    for j in xrange(10):
        print "-"*10
        print j
        print "-"*10
        for i,(e,f) in enumerate(zip(x_train,x_test)):
            if np.array_equal(e,f):
                print i
    b_train = fi_train.next()
    b_test = fi_test.next()
    x_train, y_train =  b_train[0], b_train[1]
    x_test, y_test = b_test[0], b_test[1]
    x_test = np.swapaxes(x_test,0,3)
    x_train = np.swapaxes(x_train,0,3)
    print "."




    # print "next()"
    # fi.next()
    # print "next()"
    # fi.next()
    # print "next()"
    # fi.next()


