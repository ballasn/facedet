"""
Create a list of file from a faceimage dataset
object to create a pylearn2 dataset object.

Note: this will create only positive example
"""

import sys
import os
import cv2
import numpy as np
from time import time
import theano.tensor as T
from theano import function
import numpy as np

from emotiw.common.datasets.faces.faceimages import FaceImagesDataset

def create_positives_sample(dataset,
                            output_file='pos_list.txt'):
    s = ''
    for i in xrange(0, len(dataset)):
        # Get the image path and its corresponding bounding box
        path = dataset.get_original_image_path(i)
        bbox = dataset.get_bbox(i)
        dataset.verify_samples(i)

        assert bbox is not None
        #print bbox

        for j in xrange(len(bbox)):
            assert len(bbox[j]) == 4
            # Write output
            x0, y0, x1, y1 = bbox[j]
            s += path + ' '+ str(x0) + ' ' + str(y0) + ' ' + str(x1) +' ' + str(y1)+'\n'

    with open(output_file, 'a') as output:
        output.write(s)
    return s

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage %s: <out_file>" % sys.argv[0])
        exit(1)


    out_file = sys.argv[1]

    ### AFW datasets
    # from emotiw.common.datasets.faces.afw_v2 import AFW
    # dataset = AFW()
    # t0 = time()
    # create_positives_sample(dataset, out_file)
    # t = time()
    # print t-t0, 's for AFW examples

    ### AFLW datasets
    from facedet.sandbox.faces.aflw import AFLW
    dataset = AFLW()
    print len(dataset)
    t0 = time()
    create_positives_sample(dataset, out_file)
    t = time()
    print t-t0, 's for AFLW examples'
