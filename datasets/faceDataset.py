import os
import math
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
                 ratio=0.8,
                 axes = ('b', 0, 1, 'c')):


        if which_set == 'train':
            self.positives =  np.load(positive_samples)
            nb_train = int(np.ceil(ratio * positives.shape[0]))
            self.positives = self.positives[0:nb_train, :]

            self.negatives =  np.load(negative_samples)
            nb_train = int(np.ceil(ratio * negatives.shape[0]))
            self.negatives = self.negatives[0:nb_train, :]
        elif which_set == 'valid':
            self.positives =  np.load(positive_samples)
            nb_train = int(np.ceil(ratio * positives.shape[0]))
            self.positives = self.positives[nb_train:self.positives.shape[0], :]


            self.negatives =  np.load(negative_samples)
            nb_train = int(np.ceil(ratio * negatives.shape[0]))
            self.negatives = self.negatives[nb_train:self.negatives.shape[0], :]

        self.nb_examples = self.positives.shape[0] + self.negatives.shape[0]
        self.which_set = which_set
        self.axes = axes


    def get_minibatch(self, cur_positives, cur_negatives,
                      minibatch_size,
                      data_specs, return_tuple):


        ### Initialize data
        x = np.zeros(self.vidShape + [lastIdx - firstIdx,],
                     dtype = "float32")
        y = np.zeros([lastIdx - firstIdx, self.nbTags],
                     dtype="float32")


        ### Get number of positives and negatives examples
        nb_pos = int(np.random.rand() * minibatch_size)
        nb_neg = minibatch_size - nb_pos

        ### Check boundary FIXME verify
        ### nb_examples must be divisible by minibatch_size
        if (cur_negatives + nb_neg >= self.negatives.shape[0]):
            nb_neg = self.negatives.shape[0] - cur_negatives
            nb_pos =  minibatch_size - nb_neg
        if (cur_positives + nb_pos >= self.positives.shape[0]):
            nb_pos = self.positives.shape[0] - cur_positives
            nb_neg =  minibatch_size - nb_pos

        ### Fill minibatch
        x = self.positives[cur_positives:cur_positives+nb_pos, :]
        y[cur_positives:cur_positives+nb_pos, :] = 1
        x = self.negatives[cur_negatives:cur_negatives+nb_neg, :]
        y[cur_negatives:cur_negatives+nb_neg, :] = 0


        cur_positives += nb_pos
        cur_negatives += nb_neg
        return x, y, cur_positives, cur_negatives



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
        self._dataset_size = len(dataset.nb_examples)

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

        self._num_pos = self._dataset.positives.shapes[0]
        self._num_neg = self._dataset.negatives.shapes[0]

        self._return_tuple = return_tuple
        self._data_specs = data_specs

        self.num_examples = self._dataset_size # Needed by Dataset interface
        print self.num_examples


    def __iter__(self):
        return self

    def next(self):
        if self._cur_pos >= self._num_pos and self._cur_neg >= self._num_neg:
            print self.num_example
            print self._num_batches
            raise StopIteration()
        else:
            data,
            self._cur_pos,
            self._cur_neg =  self._dataset.get_minibatch(self._cur_pos,
                                                         self._cur_neg
                                                         self._batch_size,
                                                         self._data_specs,
                                                         self._return_tuple)
            return data







