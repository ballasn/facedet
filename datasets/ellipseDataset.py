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
from math import sqrt




class ellipseDataset(dataset.Dataset):

    mapper = {'train': 0, 'valid': 1}

    def __init__(self,
                 feats,
                 labels,
                 which_set,
                 ratio=0.8,
                 batch_size=128):
        """
        Instantiates a handle to the ellipse dataset
        """
        # Make pointer to data we will slice directly from disk
        self.feats = np.load(feats, mmap_mode='r')
        self.labels = np.load(labels, mmap_mode='r')

        if which_set == 'train':
            nb_train = int(np.ceil(ratio * self.feats.shape[0]))
            self.feats = np.array(self.feats[0:nb_train, :])
            self.labels = np.array(self.labels[0:nb_train, :])
            print "train:", self.feats.shape, self.labels.shape
        elif which_set == 'valid':
            nb_train = int(np.ceil((1.0 - ratio) * self.feats.shape[0]))
            self.feats = np.array(self.feats[-nb_train:, :])
            self.labels = np.array(self.labels[-nb_train:, :])
            print "valid:", self.feats.shape, self.labels.shape

        # duplicate last line to have nb_pos, nb_neg divisible by batch_size
        # batch_size = batch_size / 2
        self.nb_feats = self.feats.shape[0]
        print self.feats.shape, self.labels.shape
        if (self.nb_feats % batch_size != 0):
            to_add = batch_size - self.nb_feats % batch_size
            for k in range(0, to_add):
                self.feats = np.append(self.feats,
                                       self.feats[-1, :].reshape(1, self.feats.shape[1]), axis=0)
                self.labels = np.append(self.labels,
                                        self.labels[-1, :].reshape(1, self.labels.shape[1]), axis=0)


        self.nb_examples = self.feats.shape[0]
        self.which_set = which_set

    def get_minibatch(self, cur,
                      minibatch_size,
                      data_specs, return_tuple):

        # Initialize data
        x = np.zeros([minibatch_size, self.feats.shape[1]], dtype="float32")
        y = np.zeros([minibatch_size, self.labels.shape[1]], dtype="float32")

        # Get data
        start = minibatch_size * cur
        end = minibatch_size * (cur + 1)
        x[:, :] = self.feats[start:end, :]
        y[:, :] = self.labels[start:end, :]

        return (x, y)



    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):

        return EllipseIterator(self, batch_size, num_batches,
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
        return self.feats

    def get_num_examples(self):
        return self.feats.shape[0]


def load_list(filename):
    id_list = []
    with open(filename, 'r') as fd:
        for line in fd:
            id_list.append(line)
    return id_list


class EllipseIterator:

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
        self._cur = 0
        self.stochastic = False
        self._return_tuple = return_tuple
        self._data_specs = data_specs

        self.num_examples = self._dataset_size # Needed by Dataset interface
        print self.num_examples


    def __iter__(self):
        return self

    def next(self):
        if self._cur >= self._num_batches:
            print "stopIteration :",
            print self.num_examples, 'ex',
            print self._num_batches, 'batches'
            raise StopIteration()
        else:
            data= self._dataset.get_minibatch(self._cur,
                                              self._batch_size,
                                              self._data_specs,
                                              self._return_tuple)
            self._cur +=1
            return data


