import os
import math
import cv2
import warnings
import numpy as np
import numpy.ma as ma
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

import os.path


class faceDataset(dataset.Dataset):

    def __init__(self,
                 positive_files,
                 negative_files,
                 which_set,
                 ratio=0.8,
                 batch_size=128,
                 ## Unhandled for now
                 axes=('b', 0, 1, 'c')):
        """
        Instantiates a handle to the face dataset
        -----------------------------------------
        positive_samples : path to the npy file of + samples
        negative_samples : path to the npy file of - samples
        The current ratio is 0.8 => train 80%, valid 20%

        WARNING: Data must be square image in RGB
        """

        self.__dict__.update(locals())

        self.positives = []
        self.negatives = []

        assert len(positive_files) == len(negative_files)
        self.load(positive_files, self.positives)
        self.load(negative_files, self.negatives)

        ### Assert that all loaded data have the same number of samples
        assert len(self.positives) > 0
        nb_pos = self.positives[0].shape[0]
        for i in xrange(1, len(self.positives)):
            assert self.positives[i].shape[0] == nb_ex
        assert len(self.negatives) > 0
        nb_neg = self.negatives[0].shape[0]
        for i in xrange(1, len(self.negatives)):
            assert self.negatives[i].shape[0] == nb_ex

        ### Compute img_shape, assuming square images in RGB
        self.img_shape = []
        for i in xrange(0, len(self.positives)):
            size = int(sqrt(self.positives[0].shape[1] / 3))
            cur_img_shape = [size, size, 3]
            self.img_shape.append(cur_img_shape)


        ### Split the set into a train/valid set
        nb_train_pos = int(np.ceil(ratio * nb_pos))
        nb_test_pos  = int(np.ceil(ratio * nb_neg))
        if which_set == 'train':
            self.pos_idx = 0, nb_train_pos
            self.neg_idx = 0, nb_train_neg
        elif which_set == 'valid':
            self.pos_idx = nb_train_pos, nb_pos
            self.neg_idx = nb_train_neg, nb_neg

        ### Make the number of max examples divisible by batch/2
        ### for implementation issues
        nb_pos = self.pos_idx[1] - self.pos_idx[0]
        nb_neg = self.neg_idx[1] - self.neg_idx[0]
        self.pos_idx[1] -= nb_pos % (batch_size/2)
        self.neg_idx[1] -= nb_neg % (batch_size/2)

        ### Print stats
        self.nb_pos = self.pos_idx[1] - self.pos_idx[0]
        self.nb_neg = self.neg_idx[1] - self.neg_idx[0]
        print "Nb positive", which_set, "examples:", nb_pos
        print "Nb negative", which_set, "examples:", nb_neg

    def load_data(self, filenames, res_lst):
        for f in filename:
            if path.splitext(f)[1] == ".hdf":
                data = tables.openFile(f, mode="r")
                res_lst.append(data.getNode('/', "denseFeat"))
            elif path.splitext(f)[1] == ".npy":
                data = np.load(f)
                res_lst.append(data)
            else:
                print "Invalid data format"
                exit(1)


    def get_minibatch(self,
                      pos_id,
                      neg_id,
                      batch_size,
                      data_specs, return_tuple):

        assert batch_size % 2 == 0
        ### Negative/Positive batch split
        split = batch_size / 2

        x = []
        for i in len(self.positives):
            x_cur = np.zeros([batch_size, self.positives[i].shape[1]],
                             dtype="float32")
            ### Fetch positives
            idx = self.pos_idx[0] + split * pos_id
            x_cur[0:split, :] = self.positives[i][idx, idx+split * pos_id, :]
            ### Fetch negatives
            idx = self.neg_idx[0] + split * neg_id
            x_cur[split:batch_size, :] = self.negatives[i][idx, idx+split * neg_id, :]
            ### Resize into b, 0, 1, c
            x_cur = np.reshape(x_cur, [minibatch_size] + self.img_shape)
            for j in xrange(0, nb_pos):
                cv2.imshow("positif " + str(j),
                           np.asarray(x_cur[j], dtype=np.uint8))
            for j in xrange(nb_pos, x.shape[0]):
                cv2.imshow("negatif " + str(j),
                           np.asarray(x_cur, dtype=np.uint8))
            cv2.waitKey(0)
            ### Transform into b c 0 1 FIXME handle other axes
            x_cur = np.transpose(x_cur, (0, 3, 1, 2))
            x.append(x_cur)

        ### Fetch labels
        ### Initialize data
        y = np.zeros([batch_size, 2], dtype="float32")
        y[0:split, 0] = 1
        y[split:batch_size, 1] = 1

        #print "return batch", cur_negatives, cur_negatives
        if data_specs == None or len(data_specs[1]) > 1:
	  return (tuple(x), y)
	else:
	  return (tuple(x),)


    def iterator(self, mode=None,
                 batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):

        ### Do the delim here
        return FaceIterator_cascade(self, batch_size, num_batches,
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



class FaceIterator_cascade:

    def __init__(self, dataset,
                 batch_size,
                 num_batches=None,
                 data_specs=False, return_tuple=False, rng=None):

        self._dataset = dataset
        self._dataset_size = dataset.get_num_examples()

        assert batch_size is not None

        # Validate the inputs
        assert dataset is not None
        if batch_size is None:
            raise ValueError("FaceIterator_cascade require batch_size")
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be divisible by two")

        if rng is None:
            self._rng = random.Random(1)
        else:
            self._rng = rng

        self._batch_size = batch_size
        self._pos_num = self._dataset.nb_pos / (batch_size / 2)
        self._neg_num = self._dataset.nb_neg / (batch_size / 2)
        self._stop_pos = self._pos_num >= self._neg_num

        self._cur_pos = 0
        self._cur_neg = 0

        self._return_tuple = return_tuple
        self._data_specs = data_specs
        self.num_examples = self._dataset_size # Needed by Dataset interface
        print self.num_examples


    def __iter__(self):
        return self

    def next(self):
        if self._cur_pos >= self._num_pos:
            if self._stop_pos:
                print "stopIteration :"
                print self.num_examples
                print self._num_batches
                raise StopIteration()
            else:
                self._cur_pos = 0
        if self._cur_neg >= self._num_neg:
            if not self._stop_pos:
                print "stopIteration :"
                print self.num_examples
                print self._num_batches
                raise StopIteration()
            else:
                self._cur_neg = 0

        data = self._dataset.get_minibatch(self._cur_pos,
                                           self._cur_neg,
                                           self._batch_size,
                                           self._data_specs,
                                           self._return_tuple)
        self._cur_pos += 1
        self._cur_neg += 1
        return data



