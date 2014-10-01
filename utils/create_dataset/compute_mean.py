import sys
import os
import cv2
import numpy as np
from time import time
import numpy as np
from random import shuffle

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: <positive> <negative> out_mean" % sys.argv[0])
        sys.exit(1)


    pos = np.load(sys.argv[1])
    neg = np.load(sys.argv[2])

    mean = (np.sum(pos, axis=0) + np.sum(neg, axis=0))/ float(pos.shape[0] +neg.shape[0])

    np.save(sys.argv[3], mean)






