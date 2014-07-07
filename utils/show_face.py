import sys
import os
import cv2
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage %s: dataset nb" % sys.argv[0])
        exit(1)

    img_shape = [48, 48, 3]


    dataset = np.load(sys.argv[1])
    sample = dataset[int(sys.argv[2]), :]

    print sample

    sample = np.reshape(sample, img_shape).astype(np.uint8)

    cv2.imshow('image', sample)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

