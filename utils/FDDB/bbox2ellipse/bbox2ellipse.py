import sys
import cv2
from os.path import join
import numpy as np


base_dir = '/data/lisa/data/faces/FDDB/'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "usage %s : <in>" % sys.argv[0]
        sys.exit(2)


    with open(sys.argv[1]) as f:
        for line in f:
            im_id = line[:-1]
            image_file = join(base_dir, line[:-1]+".jpg")
            print im_id

            img = cv2.imread(image_file)
            nb = int(f.next()[:-1])
            print nb
            # Loop over detections
            for i in xrange(nb):
                row = f.next()[:-1]
                row = row.split(' ')
                coords = [int(float(r)) for r in row[:-1]]
                score = float(row[-1])

                cv2.rectangle(img, (coords[0], coords[1]),
                              (coords[0]+coords[2], coords[1]+coords[3]),
                              (0, 255, 0), 2)


                x0 = coords[0]
                x1 = coords[0] + coords[2]
                y0 = coords[1]
                y1 = coords[1] + coords[3]

                ra = (x1 - x0) / 2
                rb = (y1 - y0) / 2
                theta = 0
                cx = x0 + ra
                cy = y0 + rb

                print float(ra), float(rb), float(theta), float(cx), float(cy), float(score)
                cv2.ellipse(img, (int(cx),int(cy)),
                            (int(ra), int(rb)),
                            int(theta),
                            0, 360, (0, 0, 255), 2)

                #cv2.imshow(image_file[len(base_dir):], img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
