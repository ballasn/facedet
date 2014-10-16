#####
# Script computing the recall on FDDB
# eg the percentage of faces that are inside the returned bounding boxes
# This is useful for the first level of the cascade

import cv2
from os.path import join
import numpy as np
import glob


def point_on_ellipse(ra, rb, theta, cx, cy, t):
    # return the coords of the point on ellipse param by t
    # ellipse defined as : ra, rb, theta, cx, cy
    # theta, t in radians
    x = cx + ra*np.cos(t)*np.cos(theta) - rb*np.sin(t)*np.sin(theta)
    y = cy + ra*np.cos(t)*np.sin(theta) + rb*np.sin(t)*np.cos(theta)
    return (x, y)


def ellipse_extrema(ra, rb, theta, cx, cy):
    # get the parameters at extreme points
    tx = np.arctan(-float(rb) / float(ra) * np.tan(theta))
    ty = np.arctan(float(rb) / float(ra) * 1.0/np.tan(theta))

    # get the extreme values
    mx1 = point_on_ellipse(ra, rb, theta, cx, cy, tx)[0]
    my1 = point_on_ellipse(ra, rb, theta, cx, cy, ty)[1]
    mx2 = 2 * cx - mx1
    my2 = 2 * cy - my1

    return (min(mx1, mx2), max(mx1, mx2), min(my1, my2), max(my1, my2))


def ellipse_in_rect(extremes, r):
    # theta is the angle (major axis, horizontal axis)
    # rectangle defined as : x, y, w, h
    (mx, Mx, my, My) = extremes
    return mx > r[0] and Mx < r[0]+r[2] and my > r[1] and My < r[1]+r[3]


if __name__ == '__main__':
    detect_dir = '../FDDB/output/'
    base_dir = '/data/lisa/data/faces/FDDB/'
    annot_dir = base_dir+'/FDDB-folds/'
    detect_files = glob.glob(detect_dir+'*-out.txt')
    detect_files.sort()
    annot_files = glob.glob(annot_dir+'*ellipse*')
    annot_files.sort()
    total = 0
    included = 0
    for a_file, d_file in zip(annot_files, detect_files):
        print a_file, d_file
        with open(a_file, 'r') as a_f:
            # Removes \n at EOLs
            annot = a_f.read().splitlines()

        with open(d_file, 'r') as f:
            cur = 0
            for line in f:
                image_file = join(base_dir, line[:-1]+".jpg")
                cur += 1
                img = cv2.imread(image_file)
                nb = int(f.next()[:-1])
                nb_a = int(annot[cur])
                total += nb_a
                cur += 1

                # Getting the list of boxes
                boxes = []
                for i in range(nb):
                    row = f.next()[:-1]
                    row = row.split(' ')
                    boxes.append([int(float(r)) for r in row[:-1]])

                # Getting the list of ellipses
                ellipses = []
                for j in xrange(nb_a):
                    row = annot[cur].split(' ')
                    [ra, rb, theta, cx, cy] = [float(r) for r in row[:-2]]
                    ellipses.append(ellipse_extrema(ra, rb, theta, cx, cy))
                    cur += 1

                for b in boxes:
                    for ind, e in enumerate(ellipses):
                        if e is None:
                            continue
                        elif ellipse_in_rect(e, b):
                            ellipses[ind] = None
                    ellipses = [x for x in ellipses if x is not None]
                    if ellipses == []:
                        break

                # Compute number of included ellipses
                included += nb_a - len(ellipses)

    print 'Faces included in boxes :', included, '/', total

