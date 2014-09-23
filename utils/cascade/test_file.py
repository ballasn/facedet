import numpy as np
import cv2
from image_processing import cut_in_squares, reconstruct, reconstruct_pred_map
from time import time
import theano.tensor as T
from theano import function
import cPickle as pkl
from nms import nms
from rois import get_rois, correct_rois, rois_to_slices


def testReconstruct():
    # Define image and model
    img_file = "/data/lisa/data/faces/FDDB/2002/08/11/big/img_276.jpg"
    model_file = '../../exp/2layers_16.pkl'
    img = cv2.imread(img_file)
    print img_file
    print model_file
    print img.shape, img.dtype

    # Define square parts and check construction
    c = cut_in_squares(img, 1, 4, 200)
    t0 = time()
    img2 = reconstruct(img, c, 1, 4, 200)
    t = time()
    print t - t0, 'seconds'
    print 'img_reconstructed', img2.shape, 'original',  img.shape
    print 'good reconstruction :', np.array_equal(img, img2)

    # Formatting the mini batch
    ex = np.reshape(c, (c.shape[0] * c.shape[1], c.shape[2], c.shape[3],
                    c.shape[4]))
    ex = np.transpose(ex, (3, 1, 2, 0))
    print 'squares', c.shape
    print 'examples', ex.shape

    with open(model_file, 'r') as m_f:
        model = pkl.load(m_f)

    # Compile network prediction function
    x = T.tensor4("x")
    predict = function([x], model.fprop(x))

    lab = predict(ex)
    print 'labels', lab.shape
    p = reconstruct_pred_map(lab, c, img, 1, 16)
    print 'pred_map', p.shape
    best_match = np.unravel_index(p.argmax(), p.shape)
    print 'best_match', best_match, p[best_match]
    bm = [best_match[0] + 16, best_match[1] + 16]
    print p[-10:-1, -10:-1]
    cv2.rectangle(img, tuple(best_match), tuple(bm), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def testNMS():
    """
    Toy example to check that NMS is working
    """
    maps = {}
    maps[1] = np.array([[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 2, 1, 1],
                        [0, 1, 0, 3]])
    maps[0.5] = np.array([[2, 0],
                          [0, 0]])
    for m in maps:
        print maps[m].shape
        print maps[m]
    t0 = time()
    map_ = nms(1, 1, maps)
    t = time()
    for m in map_:
        print map_[m].shape
        print map_[m]
    print t-t0, "seconds for nms"


def test_rois(p=0.2):
    # Example image
    res = {}
    res[0.5] = np.random.rand(105, 105)
    res[0.8] = np.random.rand(173, 173)
    for m in res:
        print res[m].shape
        print res[m]

    # Perform NMS
    t0 = time()
    # nms(size, stride, maps)
    map_ = nms(16, 1, res, acc_prob=p)
    t = time()
    for m in map_:
        print map_[m].shape
        print map_[m]

    # Now getting RoIs
    print t-t0, "seconds for nms"
    rois0 = get_rois(map_, 16, 1, enlarge_factor=0.0)
    rois1 = get_rois(map_, 16, 1, enlarge_factor=5.5)
    rois_c = correct_rois(rois1, img.shape)
    rval = rois_to_slices(rois_c)
    for i in range(len(rois1)):
        if rois1[i][0, 0] < 0 or rois1[i][0, 1] < 0:
            print rois1[i]
            print "without enlarging"
            print rois0[i]
            print "with enlarging"
            print rois1[i]
            print "corrected"
            print rois_c[i]
            break

    print rval
