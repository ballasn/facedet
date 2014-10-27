import sys
import cv2
from os.path import join
import numpy as np

import theano.tensor as T
from theano import function

import cPickle as pkl


base_dir = '/data/lisa/data/faces/FDDB/'

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "usage %s : <in> <model> <reg>" % sys.argv[0]
        sys.exit(2)

    patch_size = 48
    feats_size = 2*2*512

    ### Comput models and regressor
    with open(sys.argv[2], 'r') as m_f:
        model = pkl.load(m_f)
        #print model
        ## Sigmoid/Softmax
        model.layers.pop()
        ## FC connected layers
        model.layers.pop()
    x = T.tensor4('x')
    extract_feats = function([x], model.fprop(x))
    with open(sys.argv[3], 'r') as m_r:
        model = pkl.load(m_r)
        #print model
    v = T.matrix('v')
    reg = function([v], model.fprop(v))



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


                ### Extract feats
                [x, y, w, h] = coords
                feat = np.zeros((1, feats_size+2),dtype=np.float32)
                feat[0, 0:2] = [w, h]
                patch = img[int(max(0, int(y))):
                                int(min(img.shape[0], int(y+h))),
                            int(max(0, int(x))):
                                int(min(img.shape[1], int(x+w))), :]
                #print x,y, w, h, patch.shape
                patch = cv2.resize(patch, (patch_size, patch_size),
                                   interpolation=cv2.INTER_CUBIC)
                patch = np.reshape(patch, list(patch.shape)+[1])
                ## C01B
                patch = np.transpose(patch, (2, 0, 1, 3))
                ### BC01
                patch = np.transpose(patch, (3, 0, 1, 2))
                #print patch.shape
                tmp = extract_feats(patch)
                feat[0:, 2:] = tmp.reshape(1,
                                           tmp.shape[0]*tmp.shape[1]*tmp.shape[2]*tmp.shape[3])

                ### Apply regressors
                el_reg = reg(feat)

                ### Compute the ellipse
                # x0 = coords[0]
                # x1 = coords[0] + coords[2]
                # y0 = coords[1]
                # y1 = coords[1] + coords[3]
                # ra = (x1 - x0) / 2
                # rb = (y1 - y0) / 2
                # theta = 0
                # cx = x0 + ra
                # cy = y0 + rb


                [ra, rb, theta, cx, cy] = el_reg[0]
                cx += x
                cy += y
                print float(ra), float(rb), float(theta), float(cx), float(cy), float(score)
                cv2.ellipse(img, (int(cx),int(cy)),
                            (int(rb), int(ra)),
                            int(theta),
                            0, 360, (0, 0, 255), 2)



            #cv2.imshow(image_file[len(base_dir):], img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
