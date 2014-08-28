import os
import math
import cv2
import warnings
import numpy as np
import random
import thread
import time
import sys
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
from math import sqrt




class faceDataset(dataset.Dataset):

    mapper = {'train': 0, 'valid': 1}

    def __init__(self,
                 positive_samples,
                 negative_samples,
                 which_set,
                 ratio=0.8,
                 batch_size=128,
                 axes=('c', 0, 1, 'b')):
        """
        Instantiates a handle to the face dataset
        -----------------------------------------
        positive_samples : path to the npy file of + samples
        negative_samples : path to the npy file of - samples
        The current ratio is 0.8 => train 80%, valid 20%
        """

        ### Load data
        self.positives_h5file = tables.openFile(positive_samples, mode="r")
        self.positives = self.positives_h5file.getNode('/', "denseFeat")

        self.negatives_h5file = tables.openFile(negative_samples, mode="r")
        self.negatives = self.negatives_h5file.getNode('/', "denseFeat")
        print "done opening hdf5 files"
        if which_set == 'train':
            nb_train = int(np.ceil(ratio * self.positives.shape[0]))
            self.positives_shape = (nb_train, self.positives.shape[1])
            nb_train = int(np.ceil(ratio * self.negatives.shape[0]))
            self.negatives_shape = (nb_train, self.negatives.shape[1])
            self.start_pos = 0
            self.start_neg = 0

        elif which_set == 'valid':
            nb_train = int(np.ceil((1.0-ratio) * self.positives.shape[0]))
            start_ = int(np.ceil(ratio * self.positives.shape[0]))
            self.positives_shape = (nb_train, self.positives.shape[1])
            # We don't slice arrays to reduce memory usage]
            # start indicates the index at which starts the validation set
            self.start_pos = start_
            nb_train = int(np.ceil((1.0-ratio) * self.negatives.shape[0]))
            start_ = int(np.ceil(ratio * self.positives.shape[0]))
            self.negatives_shape = (nb_train, self.negatives.shape[1])
            self.start_neg = start_

        print "done defining positives and negatives"


        ### nb_pos and nb_neg must be divisible by batch_size / 2
        #batch_size = batch_size / 2
        self.nb_pos = self.positives_shape[0]
        self.nb_neg = self.negatives_shape[0]

        self.nb_pos = self.nb_pos - self.nb_pos % batch_size
        self.nb_neg = self.nb_neg - self.nb_neg % batch_size
        # self.img_shape = [48, 48, 3]
        # Compute img_shape, assuming square images in RGB
        size = int(sqrt(self.positives_shape[1] / 3))
        self.img_shape = [size, size, 3]
        print "image shape :", self.img_shape
        self.nb_examples = self.nb_pos + self.nb_neg
        self.which_set = which_set
        self.axes = axes
        print "positives valid shape", self.positives_shape
        print "negatives valid shape", self.negatives_shape


    def get_minibatch(self, cur_positives, cur_negatives,
                      minibatch_size,
                      data_specs, return_tuple):

        # Initialize data
        x = np.zeros([minibatch_size, self.positives_shape[1]],
                     dtype="float32")
        y = np.zeros([minibatch_size, 2],
                     dtype="float32")

        # Get number of positives and negatives examples
        # nb_pos = int(0.5 * minibatch_size)
        nb_pos = int(np.random.rand() * minibatch_size)
        nb_neg = minibatch_size - nb_pos

        # nb_examples must be divisible by minibatch_size
        too_many_neg = cur_negatives + nb_neg >= self.nb_neg
        too_many_pos = cur_positives + nb_pos >= self.nb_pos
        # In this case, we need to res
        if too_many_neg and too_many_pos:
            nb_neg = self.nb_neg - cur_negatives
            nb_pos = minibatch_size - nb_neg
            cur_pos_ = self.start_pos
            cur_neg_ = cur_negatives + self.start_neg


        elif too_many_neg:
            nb_neg = self.nb_neg - cur_negatives
            nb_pos = minibatch_size - nb_neg
            cur_pos_ = cur_positives + self.start_pos
            cur_neg_ = cur_negatives + self.start_neg

        elif too_many_pos:
            nb_pos = self.nb_pos - cur_positives
            nb_neg = minibatch_size - nb_pos
            cur_pos_ = cur_positives + self.start_pos
            cur_neg_ = cur_negatives + self.start_neg
        else:
            # Setting the real starting point
            cur_pos_ = cur_positives + self.start_pos
            cur_neg_ = cur_negatives + self.start_neg

        assert nb_pos + nb_neg == minibatch_size

        # Fill minibatch
        # cur_pos_ represent the real index on the array
        # whereas cur_positives is the one on the subest of the corresponding
        # class in the dataset (train or valid)

        try:
            x[0:nb_pos, :] = self.positives[cur_pos_: cur_pos_ + nb_pos, :]
        except ValueError:
            print "nb_pos",nb_pos
            print "size of self.pos",
            print self.positives[cur_pos_: cur_pos_ + nb_pos, :].shape
            sys.exit(1)
        y[0:nb_pos, 0] = 1
        try:
            x[nb_pos:nb_pos+nb_neg, :] = self.negatives[cur_neg_: cur_neg_ + nb_neg, :]
        except ValueError:
            print "nb_neg", nb_neg
            print "size of self.neg",
            print self.negatives[cur_neg_: cur_neg_ + nb_neg, :].shape
            sys.exit(1)
        y[nb_pos:nb_pos+nb_neg, 1] = 1

        x = np.reshape(x, [minibatch_size] + self.img_shape)
        x = np.swapaxes(x, 0, 3)
        cur_positives += nb_pos
        cur_negatives += nb_neg

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
        # To differentiate train and valid
        self._start_pos = dataset.start_pos
        self._start_neg = dataset.start_neg
        self._cur_pos = 0
        self._cur_neg = 0
        self.stochastic = False

        self._num_pos = self._dataset.positives_shape[0]
        self._num_neg = self._dataset.negatives_shape[0]

        self._return_tuple = return_tuple
        self._data_specs = data_specs

        self.num_examples = self._dataset_size # Needed by Dataset interface
        print self.num_examples

    def __iter__(self):
        return self

    def next(self):
        if self._cur_pos >= self._num_pos and self._cur_neg >= self._num_neg:
            print "stopIteration :",
            print self.num_examples,"ex",
            print self._num_batches,"batches"
            raise StopIteration()
        else:
            data, self._cur_pos, self._cur_neg = \
                self._dataset.get_minibatch(self._cur_pos,
                                            self._cur_neg,
                                            self._batch_size, self._data_specs,
                                            self._return_tuple)
            return data


if __name__ == "__main__":
    pos = "/data/lisatmp3/chassang/facedet/96/pos96.hdf"
    neg = "/data/lisatmp3/chassang/facedet/96/neg96.hdf"

    print "instantiating datasets"
    fd_train = faceDataset(pos, neg, "train")
    fd_test = faceDataset(pos, neg, "valid")
    print "Done, now the iterator"
    print type(fd_train)
    print type(fd_test)
    fi_train = FaceIterator(dataset=fd_train, batch_size=128)
    fi_test = FaceIterator(dataset=fd_test, batch_size=128)
    print fi_test._cur_pos, fi_test._cur_neg
    print fi_train._cur_pos, fi_train._cur_neg
    b_train = fi_train.next()
    b_test = fi_test.next()
    print b_train[0].shape
    print b_train[1].shape
    x_train, y_train = b_train[0], b_train[1]
    x_test, y_test = b_test[0], b_test[1]
    print "train xy",
    print x_train.shape,
    print y_train.shape
    print "test xy",
    print x_test.shape,
    print y_test.shape


    x_test = np.swapaxes(x_test, 0, 3)
    x_train = np.swapaxes(x_train, 0, 3)
    print "x test, train swapped",
    print x_test.shape,
    print x_train.shape

    eq = True
    fi_train._cur_pos = 500000
    fi_train._cur_neg = 500000
    for i, e in enumerate(fi_train):
        if i%100 ==0:
            sys.stdout.write('\r'+str(i))
            sys.stdout.flush()
        if e[0].shape[3] != 128:
            print e[0].shape, i
            sys.exit(1)
    print ""
    if not eq:
        print "train and valid batches were completely different"


    b_train = fi_train.next()
    b_test = fi_test.next()
    x_train, y_train =  b_train[0], b_train[1]
    x_test, y_test = b_test[0], b_test[1]
    x_test = np.swapaxes(x_test,0,3)
    x_train = np.swapaxes(x_train,0,3)
    print "."

