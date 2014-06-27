from emotiw.common.datasets.faces.aflw import AFLW
import numpy as np
from faceimages import FaceImagesDataset
import os.path
import random


class AFLWrapper(AFLW):
    """
    Wrapper for AFLW dataset
    """
    def __init__(self):
        # super __init__ uses the .db file
        super(AFLWrapper, self).__init__()
        # TODO : write the __init__

    def load_data(self):
        """
        Loading the data in the correct form
        """
        raise NotImplementedError()

    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):
        return AFLWIterator(self,batch_size,num_batches,
                             data_specs, return_tuple, rng)
##################
###  ITERATOR  ###
##################

class AFLWIterator():

    def __init__(self, dataset=None, batch_size=1442,
                 num_batches=None, data_specs=None,
                 return_tuple=False, rng=None):
        self._dataset = dataset
        self._dataset_size = len(dataset.y)
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
        self._next_batch_no = 0
        self._batch_order = range(self._num_batches)
        self._return_tuple = return_tuple
        self._data_specs = data_specs
        self.num_examples = self._dataset_size
        self.num_example = self.num_examples

    def __iter__(self):
        return self

    def next(self):
        if self._next_batch_no >= self._num_batches:
            print self.num_example
            print self._num_batches
            raise StopIteration()
        else:
            # Determine minibatch start and end idx
            first = self._batch_order[self._next_batch_no] * self._batch_size
            if first + self._batch_size > self._dataset_size:
                last = self._dataset_size
            else:
                last = first + self._batch_size
            data = self._dataset.get_minibatch(first, last)
                                 #self._batch_order[self._next_batch_no],
                                 #self._batch_size,
                                 #self._data_specs,
                                 #self._return_tuple)
            self._next_batch_no += 1
            return data

if __name__=="__main__":
    print "-"*30
    d = AFLWrapper()
    print d,len(d)
    print d[0]
    i= d[0].original_image
    mat = np.asarray(i[:,:])
    print type(mat)
    print mat[0][:4]
    print "-"*30
