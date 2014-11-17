import sys
import os
import cv2
import csv


def IoU(a, b):
    """
    Return the intersection over union

    a and b must be in the form:
    a = [x0, y0, w, h]
    b = [x1, y1, w1, h1]

    """
    assert len(a) == 4 and len(b) == 4
    assert a[2] >= 0 and a[3] >= 0
    assert b[2] >= 0 and b[3] >= 0

    ax0 = a[0]
    ay0 = a[1]
    ax1 = a[0] + a[2]
    ay1 = a[1] + a[3]

    bx0 = b[0]
    by0 = b[1]
    bx1 = b[0] + b[2]
    by1 = b[1] + b[3]


    x_inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    y_inter = max(0, min(ay1, by1) - max(ay0, by0))
    intersection_area =  x_inter * y_inter

    union_area = a[2] * a[3] +  b[2] * b[3] - intersection_area

    print a, b
    print "iou:", intersection_area, union_area, intersection_area / union_area
    return intersection_area / union_area

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage %s: <ubi_annot> <our_annot> [show]" % sys.argv[0])
        sys.exit(1)

    ubi_file = sys.argv[1]
    cascade_annot = sys.argv[2]
    show = 0
    if len(sys.argv) >= 4:
        show = int(sys.argv[3])

    miss = 0

    with open(ubi_file, 'r') as a_f:
        # Removes \n at EOLs
        annot = a_f.read().splitlines()
    with open(cascade_annot, 'r') as f:
        cur = 0
        for line in f:
            image_file = line[:-1] # Removes trailing \m
            cur += 1
            print image_file
            img = cv2.imread(image_file)
            nb = int(f.next()[:-1])
            nb_a = int(annot[cur])
            cur += 1
            print nb
            assert nb == 1
            assert nb == nb_a

            # Get reference box
            row = annot[cur].split(' ')
            print row
            [x, y, lx, ly, score] = [int(float(r)) for r in row]
            if show == 1:
                cv2.rectangle(img,
                              (int(x), int(y)), (int(x+lx), int(y+ly)),
                              (0, 0, 255))


            # Get detected faces
            row = f.next()[:-1]
            row = row.split(' ')
            coords = [int(float(r)) for r in row[:-1]]
            score = float(row[-1])
            if show == 1:
                cv2.rectangle(img,
                              (coords[0], coords[1]),
                              (coords[0]+coords[2], coords[1]+coords[3]),
                              (0, 255, 0), 2)
                cv2.putText(img, "%.2f" % score,
                            (int(coords[0] + 5),  int(coords[1] + 10)),
                            cv2.FONT_HERSHEY_PLAIN, 0.5,
                            (0, 255, 0))
            cur += 1


            if IoU([x, y, lx, ly], coords) < 0.6:
                miss += 1
                print "Miss annotation:", image_file

            if show == 1:
                cv2.imshow(image_file, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print miss, "total miss annotation(s)"
