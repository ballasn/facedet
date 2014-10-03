import cv2
from os.path import join
import numpy as np

base_dir = '/data/lisa/data/faces/FDDB/'
file_ = './results/output/fold-01-out.txt'
with open(file_, 'r') as f:

    for line in f:
        image_file = join(base_dir, line[:-1]+".jpg")
        print image_file
        img = cv2.imread(image_file)
        nb = int(f.next()[:-1])
        print nb
        for i in xrange(nb):
            row = f.next()[:-1]
            row = row.split(' ')
            coords = [int(float(r)) for r in row[:-1]]
            print i, img.shape
            print i, coords
            score = float(row[-1])

            cv2.rectangle(img, (coords[0], coords[1]),
                               (coords[0]+coords[2], coords[1]+coords[3]),
                               (0, 255, 0), 2)
            cv2.putText(img, str(i),
                        (coords[1] + 10, coords[0] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0))

        cv2.imshow('r', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
