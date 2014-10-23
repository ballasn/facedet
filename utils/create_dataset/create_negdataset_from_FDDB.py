#import cv2
from os.path import join
import numpy as np
import sys
import cv2

base_dir = '/data/lisa/data/faces/FDDB/'


def IoM(a, b):
    """
    Return the intersection / min area

    a and b must be in the form:
    a = [x0, x1, y0, y1]
    b = [x0, x1, y0, y1] with x0 <= x1 and y0 <= y1

    """
    assert len(a) == 4 and len(b) == 4
    assert a[1] >= a[0] and b[1] >= b[0]
    assert a[3] >= a[2] and b[3] >= b[2]

    #min_area = min((a[1] - a[0]) * (a[3] - a[2]), (b[1] - b[0]) * (b[3] - b[2]))
    union_area = max(0, min(a[1], b[1]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[2], b[2]))

    return union_area #/ float(min_area)


def rnd_bounding_box(nrows, ncols, size=None):
    if size is None:
        # Get initial Position
        x0 = np.random.random_integers(0, int(np.ceil(0.6 * nrows)))
        y0 = np.random.random_integers(0, int(np.ceil(0.6 * ncols)))

        side = np.random.random_integers(0.1*min(nrows, ncols),
                                         0.4*min(nrows, ncols))
    else:
        x0 = np.random.random_integers(0, nrows - size)
        y0 = np.random.random_integers(0, ncols - size)
        side = size

    return np.array([x0, y0, x0+side, y0+side])


def convert_fddb_annotation(annot, nb_sample=-1, nb_draw = 20,
                            max_overlap = 0):

    with open(annot, 'r') as f:

        line = f.readline()
        while line != '':
            image_file = join(base_dir, line.strip() + ".jpg")
            nb_faces = int(f.readline().strip())
            #print image_file, nb_faces
            img = cv2.imread(image_file)

            ### Read all the annotation
            box = []
            for i in xrange(0, nb_faces):
                line = f.readline().strip()
                coords = line.split(' ')
                [ra, rb, theta, cx, cy] = [float(r) for r in coords[:-2]]
                theta = np.degrees(theta)
                #print image_file, cy-ra, cx-rb, cy+ra, cx+rb
                box.append([cy-ra, cx-rb, cy+ra, cx+rb])
                # cv2.rectangle(img,
                #               (int(cx-rb), int(cy-ra)),
                #               (int(cx+rb), int(cy+ra)),
                #               (0, 255, 0), 2)

            if nb_sample == -1:
                nb_to_sample = nb_faces
            else:
                nb_to_sample = nb_sample
            for i in xrange(0, nb_to_sample):
                ok = False
                j = 0
                while not ok and j < nb_draw:
                    sample = rnd_bounding_box(img.shape[0], img.shape[1])
                    ok = True
                    for b in xrange(len(box)):
                        # Inter over Union
                        score=IoM([sample[0], sample[2], sample[1], sample[3]],
                                  [box[b][0], box[b][2], box[b][1], box[b][3]])
                        ok = ok and (score == max_overlap)
                        j = j + 1

                    if ok:
                        print image_file, sample[0], sample[1], sample[2], sample[3]
            #             cv2.rectangle(img,
            #                           (int(sample[1]), int(sample[0])),
            #                           (int(sample[3]), int(sample[2])),
            #                           (255, 0, 0), 2)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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
