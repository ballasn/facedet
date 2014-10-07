import sys
import os
import cv2
import numpy as np
from time import time
import numpy as np
from random import shuffle



def read_list_int(filename):
    l = []
    with open(filename) as fp:
        for line in fp:
            l.append(int(line.strip()))

    return l

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage %s: <dataset_file> <list_size> [patch_size=16] [display_size=96]" % sys.argv[0])
        sys.exit(1)

    patch_size = 16
    show_size = 96
    if len(sys.argv) >= 4:
        patch_size = int(sys.argv[3])
    if len(sys.argv) >= 5:
        show_size = int(sys.argv[4])



    data = np.load(sys.argv[1])
    ids = read_list_int(sys.argv[2])


    for i in ids:
        img = data[i, :]
        img = np.asarray(img, dtype='uint8')
        img = np.reshape(img, (patch_size, patch_size, 3))
        img = cv2.resize(img, (show_size, show_size),
                         interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Img", img)
        cv2.waitKey(0)







