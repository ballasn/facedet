from os.path import join, isfile
import cv2
import numpy as np


def read_to_dict(annot_file, data='rect', wh=True):
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

                if data == 'ellipse':
                    # FDDB : [ra, rb, theta, cx, cy]
                    box_ = [int(float(r)) for r in row[:-2]]

                elif data == 'rect':
                    # AFLW : [x0, yi0, x1 ,y1]
                    # or [x, y, w, h]
                    box = [int(float(r)) for r in row[:-1]]
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
                    box_.append(float(row[-1]))
                annot[im].append(box_)
    return annot


def regress_dict(detections, detector, regressor, base_dir='', patch_size=96,
                 entry_size=8194):
    '''
    Regress the detections from a dict indexed by the file IDs
    '''
    detect_regressed = {}
    entry = np.zeros((1, entry_size))
    i = 0
    for im in detections:
        i += 1
        # Get the image
        im_f = join(base_dir, im)
        im_f += '.jpg'
        print '\r'+str(i),
        assert(isfile(im_f))
        img = cv2.imread(im_f)
        # Create the entry
        detect_regressed[im] = []
        for d in detections[im]:
            [x, y, w, h] = [int(e) for e in d[:-1]]
            sco = d[4]
            p = img[max(0, y):min(img.shape[0], y+h),
                    max(0, x):min(img.shape[1], x+w), :]
            p = cv2.resize(p, (patch_size, patch_size),
                           interpolation=cv2.INTER_CUBIC)
            p = np.transpose(p, (2, 0, 1))
            p = np.reshape(p, [1]+list(p.shape))
            feats = detector(p)
            entry[0, 0:2] = [w, h]
            feats = np.asarray(feats, dtype='float32')
            entry[0, 2:] = np.copy(np.reshape(feats,
                        feats.shape[1]*feats.shape[2]*feats.shape[3]))
            output = regressor(entry)
            output = output[0, :]
            output[0] += x
            output[1] += y
            # In case the regressor is doing something wrong
            # This will prevent the results from getting too weird
            output[2] = max(output[2], 10)
            output[3] = max(output[3], 10)
            detect_regressed[im].append(list(output)+[sco])

    return detect_regressed


def dictToFddbStyle(boxes, id_list):
    s = ''
    for idx in id_list:
        s += str(idx) + '\n'
        s += str(len(boxes[idx])) + '\n'
        for box in boxes[idx]:
            s += ' '.join(map(str, box))+'\n'
    return s

