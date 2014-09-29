import cv2
from os.path import join
import numpy as np

base_dir = '/data/lisa/data/faces/FDDB/'
file_1 = './results/output4/fold-01-out.txt'
file_2 = './results/output3/fold-01-out.txt'
with open(file_1, 'r') as f1:
    with open(file_2, 'r') as f2:

        #for line1, line2 in zip(f1, f2):
        while True:
            line1 = f1.next()
            line2 = f2.next()
            print line1
            print line2
            image_file1 = join(base_dir, line1[:-1]+".jpg")
            image_file2 = join(base_dir, line2[:-1]+".jpg")
            nb1 = int(f1.next()[:-1])
            nb2 = int(f2.next()[:-1])
            img1 = cv2.imread(image_file1)
            img2 = cv2.imread(image_file2)
            print image_file1, nb1, nb2
            for i in xrange(nb1):
                row = f1.next()[:-1]
                row = row.split(' ')
                coords = [int(float(r)) for r in row[:-1]]
                score = float(row[-1])

                cv2.rectangle(img1, (coords[0], coords[1]),
                                   (coords[0]+coords[2], coords[1]+coords[3]),
                                   (0, 255, 0), 2)
                cv2.putText(img1, str(i),
                            (coords[0] + 10, coords[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0))
            for j in xrange(nb2):
                row = f2.next()[:-1]
                row = row.split(' ')
                coords = [int(float(r)) for r in row[:-1]]
                score = float(row[-1])

                cv2.rectangle(img2, (coords[0], coords[1]),
                                   (coords[0]+coords[2], coords[1]+coords[3]),
                                   (0, 0, 255), 2)
                cv2.putText(img2, str(j),
                            (coords[0] + 10, coords[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255))

            cv2.imshow(file_1, img1)
            cv2.imshow(file_2, img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
