import sys
import cv2
from os.path import join
import numpy as np

import cPickle as pkl
import theano.tensor as T
from theano import function

base_dir = '/data/lisa/data/faces/FDDB/'


def IoU(a, b):
    """
    Return the intersection over union

    a and b must be in the form:
    a = [x0, y0, w, h]
    b = [x1, y1, w1, h1]

    """
    if not(len(a) == 4 and len(b) == 4):
        print len(a), len(b)
    if not(a[2] >= 0 and a[3] >= 0):
        print a
        exit(1)
    if not(b[2] >= 0 and b[3] >= 0):
        print b
        exit(1)

    ax0 = a[0]
    ay0 = a[1]
    ax1 = a[0] + a[2]
    ay1 = a[1] + a[3]

    bx0 = b[0]
    by0 = b[1]
    bx1 = b[0] + b[2]
    by1 = b[1] + b[3]


    intersection_area = max(0, min(ax1, bx1) - max(ax0, bx0)) *\
                        max(0, min(ay1, by1) - max(ay0, by0))
    union_area = a[2] * a[3] + b[2] * b[3] - intersection_area

    return float(intersection_area) / float(union_area)

def el2bbox(el):
    """
    Returns the enclosing bounding box of an ellipse
    el must be in the form [ra, rb, theta, cx, cy]
    """
    [ra, rb, theta, cx, cy] = el
    return [cx - rb, cy - ra, 2*rb, 2*ra]


def show_annot(img, rect, el):

    [x, y, w, h] = rect
    [xe, ye, we, he] = el2bbox(el)
    [ra, rb, theta, cx, cy] = el
    cv2.rectangle(img, (x, y), (x+h, y+h),
                  (0, 255, 0), 2)
    cv2.ellipse(img, (int(cx), int(cy)), (int(rb), int(ra)), int(theta),
                0, 360, (0, 0, 255), 2)
    cv2.rectangle(img, (int(xe), int(ye)), (int(xe+he), int(ye+he)),
                  (0, 0, 255), 2)
    cv2.imshow("img", img)


def read_to_dict(annot_file, data='fddb', wh=True):
    annot = {}
    with open(annot_file) as f:

        for line in f:
            im = line[:-1]
            annot[im] = []
            line = f.next()[:-1]
            nb = int(line)

            for i in xrange(0, nb):
                row = f.next()[:-1]
                row = row.split(' ')

                if data == 'fddb':
                    # FDDB : [ra, rb, theta, cx, cy]
                    box_ = [int(float(r)) for r in row[:-2]]

                elif data == 'aflw':
                    # AFLW : [x0, y0, x1 ,y1]
                    box = [int(float(r)) for r in row]
                    box_ = list(box)
                    box_[0] = max(0, box_[0])
                    box_[1] = max(0, box_[1])
                    if not wh:
                        box_[2] -= box_[0]
                        box_[3] -= box_[1]
                        assert any([e > 0 for e in box_])
                        #if not box_[2]==box_[3]:
                        #    print annot_file, data, wh
                        #    print 'before', box
                        #    print 'after', box_
                        #    exit(1)
                annot[im].append(box_)
    return annot


def generate_reg_data(bbox_file, annot_file,
                      save_feats, save_label,
                      extract_feats=None,
                      size_feats=0,
                      patch_size=48,
                      area_thres=0.3,
                      data='fddb'):
    # Load annotation
    annot = read_to_dict(annot_file, data=data, wh=False)
    # Load bbox
    bbox = read_to_dict(bbox_file, data=data, wh=True)
    # Compute the number of overlapped ellipse/bbox
    nb_ex = 0
    for e in bbox:
        # Loop on bbox because they don't have all images
        for an in annot[e]:
            for box in bbox[e]:
                if data == 'fddb':
                    cond = IoU(box, el2bbox(an)) > area_thres
                elif data == 'aflw':
                    #img = cv2.imread(e)
                    b = [box[1], box[0], box[3], box[2]]
                    cond = IoU(b, an) > area_thres
                    #cv2.rectangle(img, (box[1], box[0]),
                    #              (box[1]+box[3], box[0]+box[2]),
                    #              (0, 0, 255), 2)
                    #cv2.rectangle(img, (an[0], an[1]),
                    #              (an[2]+an[0], an[3]+an[1]),
                    #              (0, 255, 0), 2)
                    #cv2.imshow(e, img)
                    #cv2.moveWindow(e, 200,600)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                if cond:
                    #print an, box
                    #print 'one'
                    nb_ex += 1

    print "Found", nb_ex, "Overlapped annotation"

    # Create feats, labels matrix
    feats = np.zeros((nb_ex, 2 + size_feats))
    if data == 'fddb':
        labels = np.zeros((nb_ex, 5))
    elif data == 'aflw':
        labels = np.zeros((nb_ex, 4))
    coords = np.zeros((nb_ex, 2))

    # fills the feats, labels matrix
    cur = 0
    for im in bbox:
        for box in bbox[im]:
            for an in annot[im]:
                if data == 'fddb':
                    [x, y, w, h] = box
                    [ra, rb, theta, cx, cy] = an
                    cond = IoU(box, el2bbox(an)) > area_thres
                elif data == 'aflw':
                    [y, x, h, w] = box
                    [xa, ya, wa, ha] = an
                    cond = IoU(box, an) > area_thres
                if cond:

                    feats[cur, 0:2] = [w, h]
                    # Elipse size and center offset
                    labels[cur, :] = [xa - x, ya - y, wa, ha]
                    # Store the offset in coords
                    coords[cur, :] = [x, y]

                    if extract_feats is not None:
                        # Load img
                        if data == 'fddb':
                            img_file = join(base_dir, im + ".jpg")
                        elif data == 'aflw':
                            img_file = im
                        img = cv2.imread(img_file)

                        # Extract patch
                        patch = img[int(max(0, int(y))):
                                        int(min(img.shape[0], int(y+h))),
                                    int(max(0, int(x))):
                                        int(min(img.shape[1], int(x+w))), :]
                        patch = cv2.resize(patch, (patch_size, patch_size),
                                           interpolation=cv2.INTER_CUBIC)
                        patch = np.reshape(patch, list(patch.shape)+[1])

                        ## C01B
                        patch = np.transpose(patch, (2, 0, 1, 3))
                        ### BC01
                        patch = np.transpose(patch, (3, 0, 1, 2))
                        feat = extract_feats(patch)

                        feat = feat.reshape(1,
                                            feat.shape[0]*feat.shape[1]*feat.shape[2]*feat.shape[3])
                        feats[cur, 2:] = feat
                    cur += 1

    # write the output labels
    np.save(save_feats, feats)
    np.save(save_label, labels)
    np.save(save_label + "_coord.npy", coords)


if __name__ == '__main__':

    if len(sys.argv) < 5:
        print "usage %s : <detections> <annot> <feats_file> <labels_file> [model]" % sys.argv[0]
        print "if you used a model, please verify the feats_size in the code"
        sys.exit(2)

    patch_size = 48
    feats_size = 2*2*512
    extractfeat = None
    if len(sys.argv) == 6:
        with open(sys.argv[5], 'r') as m_f:
            model = pkl.load(m_f)

        print model
        # Sigmoid/Softmax
        model.layers.pop()
        # FC connected layers
        model.layers.pop()
        x = T.tensor4('x')
        extractfeat = function([x], model.fprop(x))



    generate_reg_data(sys.argv[1], sys.argv[2],
                      sys.argv[3], sys.argv[4],
                      extractfeat, feats_size,
                      patch_size, data='aflw')


