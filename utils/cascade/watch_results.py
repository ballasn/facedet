import cv2
from os.path import join
import numpy as np

base_dir = '/data/lisa/data/faces/FDDB/'
file_ = '../FDDB/output/fold-01-out.txt'
annot_file = base_dir+'/FDDB-folds/FDDB-fold-01-ellipseList.txt'
with open(annot_file, 'r') as a_f:
    # Removes \n at EOLs
    annot = a_f.read().splitlines()
with open(file_, 'r') as f:
    cur = 0
    for line in f:
        image_file = join(base_dir, line[:-1]+".jpg")
        cur += 1
        print image_file
        img = cv2.imread(image_file)
        nb = int(f.next()[:-1])
        nb_a = int(annot[cur])
        cur += 1
        print nb

        # Loop over detections
        for i in xrange(nb):
            row = f.next()[:-1]
            row = row.split(' ')
            coords = [int(float(r)) for r in row[:-1]]
            #print i, img.shape
            #print i, coords
            score = float(row[-1])
            '''
            cv2.rectangle(img, (coords[0], coords[1]),
                               (coords[0]+coords[2], coords[1]+coords[3]),
                               (0, 255, 0), 2)
            cv2.putText(img, str(i),
                        (coords[1] + 10, coords[0] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0))
            '''
        # Loop over annotations
        for j in xrange(nb_a):
            row = annot[cur].split(' ')
            print row
            # coords : ra, rb, theta, cx, cy, s
            [ra, rb, theta, cx, cy] = [float(r) for r in row[:-2]]
            theta = np.degrees(theta)
            cv2.ellipse(img, (int(cx),int(cy)), (int(ra), int(rb)), int(theta),
                        0, 360, (0, 0, 255), 2)
            cur += 1


        cv2.imshow(image_file[len(base_dir):], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
