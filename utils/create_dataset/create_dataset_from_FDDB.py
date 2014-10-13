#import cv2
from os.path import join
import numpy as np
import sys

base_dir = '/data/lisa/data/faces/FDDB/'




def convert_fddb_annotation(annot):

    with open(annot, 'r') as f:

        line = f.readline()
        while line != '':
            image_file = join(base_dir, line.strip() + ".jpg")
            print image_file

            nb_faces = int(f.readline().strip())
            #print image_file, nb_faces

            for i in xrange(0, nb_faces):
                line = f.readline().strip()
                print line
                coords = line.split(' ')
                [ra, rb, theta, cx, cy] = [float(r) for r in coords[:-2]]
                theta = np.degrees(theta)
                print image_file, cx-ra, cy-rb, 2*ra, 2*rb

            line = f.readline()
            # # Loop over annotations
            # for j in xrange(nb_a):
            #     row = annot[cur].split(' ')
            #     print row
            #     # coords : ra, rb, theta, cx, cy, s
            #     [ra, rb, theta, cx, cy] = [float(r) for r in row[:-2]]
            #     theta = np.degrees(theta)
            #     cv2.ellipse(img, (int(cx),int(cy)), (int(ra), int(rb)), int(theta),
            #                 0, 360, (0, 0, 255), 2)
            #     cur += 1






if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Usage %s: <annot_file>" % sys.argv[0]
        exit(1)
    convert_fddb_annotation(sys.argv[1])
