import os
import math
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
                 axes=('b', 'c', 0, 1),
                 nb_examples=[None, None]):
        """
        Instantiates a handle to the face dataset
        -----------------------------------------
        positive_samples : path to the npy file of + samples
        negative_samples : path to the npy file of - samples
        The current ratio is 0.8 => train 80%, valid 20%
        """

        # Load data
        self.positives_h5file = tables.openFile(positive_samples, mode="r")
        self.positives = self.positives_h5file.getNode('/', "denseFeat")

        self.negatives_h5file = tables.openFile(negative_samples, mode="r")
        self.negatives = self.negatives_h5file.getNode('/', "denseFeat")
        print "done opening hdf5 files"
        if nb_examples[0] is None:
            nb_examples[0] = self.positives.shape[0]
        if nb_examples[1] is None:
            nb_examples[1] = self.negatives.shape[0]

        if which_set == 'train':
            nb_train = int(np.ceil(ratio * nb_examples[0]))
            self.positives_shape = (nb_train, self.positives.shape[1])
            nb_train = int(np.ceil(ratio * nb_examples[1]))
            self.negatives_shape = (nb_train, self.negatives.shape[1])
            self.start_pos = 0
            self.start_neg = 0

        elif which_set == 'valid':
            nb_train = int(np.ceil((1.0-ratio) * nb_examples[0]))
            start_ = int(np.ceil(ratio * nb_examples[0]))
            self.positives_shape = (nb_train, self.positives.shape[1])
            # We don't slice arrays to reduce memory usage
            # start indicates the index at which starts the validation set
            self.start_pos = start_

            nb_train = int(np.ceil((1.0-ratio) * nb_examples[1]))
            start_ = int(np.ceil(ratio * nb_examples[1]))
            self.negatives_shape = (nb_train, self.negatives.shape[1])
            self.start_neg = start_

        print "done defining positives and negatives"


        ### nb_pos and nb_neg must be divisible by batch_size / 2
        #batch_size = batch_size / 2
        self.nb_pos = self.positives_shape[0]
        self.nb_neg = self.negatives_shape[0]

        self.nb_pos = self.nb_pos - self.nb_pos % batch_size
        self.nb_neg = self.nb_neg - self.nb_neg % batch_size

        # Compute img_shape, assuming square images in RGB
        size = int(sqrt(self.positives_shape[1] / 3))
        self.img_shape = [size, size, 3]
        self.nb_examples = self.nb_pos + self.nb_neg
        self.which_set = which_set
        self.axes = axes
        print "Image shape :", self.img_shape
        print "Positives shape", self.positives_shape
        print "Negatives shape", self.negatives_shape
        # Define the permutation to be applied to the mini batch
        #   to get the good format
        data_axes = ('b', 0, 1, 'c')  # The axes of the data in hdf5
        new_axes = []
        for e in self.axes:
            new_axes.append(data_axes.index(e))
        self.permutation = tuple(new_axes)  # Permutation to be applied

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

        if (cur_negatives + nb_neg >= self.nb_neg):
            nb_neg = self.nb_neg - cur_negatives
            nb_pos = minibatch_size - nb_neg


        if (cur_positives + nb_pos >= self.nb_pos):
            nb_pos = self.nb_pos - cur_positives
            nb_neg = minibatch_size - nb_pos

        # Absolute indices
        cur_pos_ = cur_positives + self.start_pos
        cur_neg_ = cur_negatives + self.start_neg

        assert nb_pos + nb_neg == minibatch_size
        assert nb_pos >= 0
        assert nb_neg >= 0

        # Writing positive examples
        x[0:nb_pos, :] = self.positives[cur_pos_: cur_pos_ + nb_pos, :]
        y[0:nb_pos, 0] = 1

        x[nb_pos: nb_pos + nb_neg, :] = self.negatives[cur_neg_: cur_neg_
                + nb_neg, :]
        y[nb_pos: nb_pos + nb_neg, 1] = 1

        # Transforming into B01C
        x = np.reshape(x, [minibatch_size] + self.img_shape)

        # C01B conversion
        #x = np.swapaxes(x, 0, 3)
        # BC01 conversion

        x = np.transpose(x, self.permutation)


        cur_positives += nb_pos
        cur_negatives += nb_neg

        return (x, y), cur_positives, cur_negatives


    def get_negseqs_minibatch(self, cur, minibatch_size,
                              data_specs, return_tuple):
        ### Initialize data
        x = np.zeros([minibatch_size, self.positives.shape[1]],
                     dtype = "float32")
        y = np.zeros([minibatch_size, 2],
                     dtype="float32")
        x[0:minibatch_size, :] = self.negatives[cur*minibatch_size:(cur + 1) * minibatch_size, :]
        x = np.reshape(x, [minibatch_size] + self.img_shape)
        x = np.swapaxes(x, 3, 0)
        y[0:minibatch_size, 1] = 1
        assert y[0,0] ==0
        x = np.transpose(x, self.permutation)
        return (x, y)



    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):

        if mode == 'negative_seq':
            print '*'*30
            print "Returning only negatives"
            print '*'*30
            return FaceIteratorNegSeq(self,
                                      batch_size, num_batches,
                                      data_specs, return_tuple, rng)
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
        self._cur_pos = 0
        self._cur_neg = 0
        self.stochastic = False

        self._num_pos = self._dataset.nb_pos
        self._num_neg = self._dataset.nb_neg
        self._return_tuple = return_tuple
        self._data_specs = data_specs

        self.num_examples = self._dataset_size # Needed by Dataset interface

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

class FaceIteratorNegSeq:

    def __init__(self, dataset=None, batch_size=None, num_batches=None,
                 data_specs=False, return_tuple=False, rng=None):

        self._dataset = dataset
        self._dataset_size = dataset.negatives_shape[0]

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

        self._batch_size = batch_size
        self._num_batches = int(num_batches)
        self._cur = 0
        self.stochastic = False
        self._return_tuple = return_tuple
        self._data_specs = data_specs

        self.num_examples = self._dataset_size # Needed by Dataset interface
        #print self.num_examples


    def __iter__(self):
        return self

    def next(self):
        if self._cur == self._num_batches:
            raise StopIteration()
        else:
            data = self._dataset.get_negseqs_minibatch(self._cur,
                                                      self._batch_size,
                                                      self._data_specs,
                                                      self._return_tuple)
            self._cur += 1
            return data

if __name__ == "__main__":
    pos = "/data/lisatmp3/chassang/facedet/96/pos96.hdf"
    neg = "/data/lisatmp3/chassang/facedet/96/neg96.hdf"

    print "instantiating datasets"
    fd_train = faceDataset(pos, neg, "train")
    fd_test = faceDataset(pos, neg, "valid")
    print "Done, now the iterator"
    while True:
        fi_train = FaceIterator(dataset=fd_train, batch_size=128)
        eq = True
        fi_train._cur_pos = 558000
        fi_train._cur_neg = 558000
        c = 0
        for i, e in enumerate(fi_train):
            c += 1
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

