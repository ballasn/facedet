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
    b = [x0, x1, w, h]

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


    intersection_area = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union_area = a[2] * a[3] +  b[2] * b[3] - intersection_area

    return intersection_area / union_area

def el2bbox(el):
    """
    Returns the enclosing bounding box of an ellipse
    el must be in the form [ra, rb, theta, cx, cy]
    """
    [ra, rb, theta, cx, cy] = el
    return [cx - ra, cy - rb, 2*ra, 2*rb]


def generate_reg_data(bbox_file, annot_file,
                      save_feats, save_label,
                      extract_feats = None,
                      size_feats = 0,
                      path_size=48,
                      area_thres = 0.3):

    # Load annotation
    annot = []
    annot_im = []
    with open(annot_file) as f:
        for line in f:
            im = line[:-1]
            annot_im.append(im)
            line =  f.next()[:-1]
            nb = int(line)
            cur = len(annot_im) - 1
            annot.append([])
            for i in xrange(0, nb):
                row = f.next()[:-1]
                row = row.split(' ')
                # [ra, rb, theta, cx, cy]
                ellipse = [float(r) for r in row[:-2]]
                annot[cur].append(ellipse)

    # Load bbox
    bbox = []
    bbox_im = []
    with open(bbox_file) as f:
        for line in f:
            im = line[:-1]
            bbox_im.append(im)
            cur = len(bbox_im) - 1
            assert annot_im[cur] == bbox_im[cur]
            nb = int(f.next()[:-1])
            bbox.append([])
            for i in xrange(0, nb):
                row = f.next()[:-1]
                row = row.split(' ')
                # [x, y, width, height]
                coords = [int(float(r)) for r in row[:-1]]
                bbox[cur].append(coords)

    ### Compute the number of overlapped ellipse/bbox
    nb_ex = 0
    for i in xrange(0, len(annot)):
        for j in xrange(0, len(bbox[i])):
            for k in xrange(0, len(annot[i])):
                if IoU(bbox[i][j], el2bbox(annot[i][k])) > area_thres:
                    nb_ex += 1


    print "Found", nb_ex, "Overlapped annotation"
    ### Create feats, labels matrix
    feats = np.zeros((nb_ex, 4 + size_feats))
    labels = np.zeros((nb_ex, 5))

    ### fills the feats, labels matrix
    cur = 0
    for i in xrange(0, len(annot)):
        for j in xrange(0, len(bbox[i])):
            for k in xrange(0, len(annot[i])):
                if IoU(bbox[i][j], el2bbox(annot[i][k])) > area_thres:
                    print cur, len(bbox[i])
                    feats[cur, 0:4] = bbox[i][j]
                    labels[cur, :] = annot[i][k]
                    if extract_feats != None:
                        ### Load img
                        image_file = join(base_dir, line[:-1]+".jpg")
                        img = cv2.imread(image_file)
                        ### Extract patch
                        [x, y, w, h] = bbox[i][j]
                        patch = img[int(max(0, int(x))):
                                        int(min(img.shape[0], int(x+w))),
                                    int(max(0, int(y))):
                                        int(min(img.shape[1], int(x+y))), :]
                        patch = np.reshape(patch, list(image.shape)+[1])
                        ## C01B
                        patch = np.transpose(patch, (2, 0, 1, 3))
                        ### BC01
                        patch = np.transpose(image, (3, 0, 1, 2))
                        feat = extract_feats(path)
                        feat = feats.reshape(1, feat.shape[0]* feat.shape[1] * feat.shape[2] * feat.shape[3])
                        feats[cur, 4:] = feat
                    cur += 1

    # write the output labels
    np.save(save_feats, feats)
    np.save(save_label, labels)


if __name__ == '__main__':

    if len(sys.argv) < 5:
        print "usage %s : <bbox> <annot> <feats> <labels> [model]" % sys.argv[0]
        print "if you used a model, please verify the feats_size in the code"
        sys.exit(2)


    feats_size = 512
    extractfeat = None
    if len(sys.argv) == 6:
        with open(sys.argv[5], 'r') as m_f:
            model = pkl.load(m_f)
        ## Sigmoid/Softmax
        model.layers.pop()
        ## FC connected layers
        model.layer.pop()
        x = T.tensor4('x')
        extractfeat = function([x], model.fprop(x))



    generate_reg_data(sys.argv[1], sys.argv[2],
                      sys.argv[3], sys.argv[4],
                      extractfeat, feats_size)


