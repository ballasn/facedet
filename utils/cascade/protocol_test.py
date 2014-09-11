import cv2
import sys
import numpy as np
from theano import function
import theano.tensor as T
import cPickle as pkl
from time import time
from nms import nms
from copy import copy

def process_image(fprop_func, image, scales):
    """
    Runs the fprop on different scales of an image
    returns a list of maps indexed like the scales
    -------------------------------------------
    fprop_func : classifier
    image_file : path to the image_file
    scales : list of scales
    """
    map_ = {}
    minibatch = {}
    for s in scales:
        img = rescale(image, s)
        #img = square(img)
        minibatch[s] = img

    map_ = apply_fprop(fprop_func, minibatch)

    return map_


def apply_fprop(fprop_func, image):
    """
    Apply a model on a image, returns the map of p(face)
    ------------------------
    fprop_func : a compiled fprop function
    image : a numpy array representing an image
    """
    if type(image) == dict:
        rval = {}
        for s in image:
            rval[s] = apply_fprop(fprop_func, image[s])
        return rval
    # Add a minibatch dim to get C01B format
    image = np.reshape(image, list(image.shape)+[1])
    image = np.transpose(image, (2, 0, 1, 3))
    rval = fprop_func(image)[0, :, :, 0]

    return rval


def rescale(image, scale):
    """
    Rescale image, returns a rescaled copy of the image
    -------------
    image : numpy array representing an image
    scale : the rescaled image has size
            scale * image_size
    """
    if scale == 1:
        return image
    sh = image.shape
    # WARNING : resize needs to receive swapped sizes to perform as we would
    # imagine
    resized_image = cv2.resize(image,
                    (max(int(sh[1] * scale), 16), max(int(sh[0] * scale), 16)),
        interpolation=cv2.INTER_CUBIC)

    resized_array = np.asarray(resized_image, dtype=image.dtype)

    return resized_array

def cut_in_squares(img, k_stride, k_shape, square_size):
    """
    Cut an image into squares to be classified independently
    Enables to process large images without mem overload
    The overlap between squares is k_shape - k_stride, eg
    we mimic taking the whole image as input
    ----------------------------------------------------
    img : numpy array representing an image
    k_stride : stride of the kernel
    k_shape : shape of the kernel
    square_size = size of the extracted images
    """
    square_stride = square_size - (k_shape - k_stride)
    # We have to deal with the image remainder
    x_squares = img.shape[0] / square_stride + 1
    y_squares = img.shape[1] / square_stride + 1

    squares = np.zeros((x_squares, y_squares, square_size, square_size, 3),
                       dtype='float32')
    for i in xrange(x_squares):
        for j in xrange(y_squares):
            init_x = i * square_stride
            init_y = j * square_stride
            # We need square_size elements
            end_x = init_x + square_size
            end_y = init_y + square_size
            if i == x_squares - 1:
                end_x = img.shape[0]
            if j == y_squares - 1:
                end_y = img.shape[1]
            squares[i, j, :end_x - init_x, :end_y - init_y, :] = img[init_x: end_x, init_y: end_y, :]

    return squares

def get_init(i, j, img, k_stride, k_shape, squares, square_size):
    """
    Returns the coords of the top-left pixel of squares[i,j]
    """
    square_stride = square_size - (k_shape - k_stride)
    init_x = i * square_stride
    init_y = j * square_stride
    return init_x, init_y

def reconstruct(img, squares, k_stride, k_shape, square_size):
    """
    Rebuilds the image from the squares
    Useful for test purpose, you can check that the squares
    were created the right way
    """
    img1 = np.zeros(img.shape)
    for i in xrange(squares.shape[0]):
        for j in xrange(squares.shape[1]):
            init_x, init_y = get_init(i, j, img,
                    k_stride, k_shape, squares, square_size)
            end_x = min(init_x + square_size, img.shape[0])
            end_y = min(init_y + square_size, img.shape[1])
            img1[init_x: end_x, init_y: end_y, :] =\
                    squares[i, j, :end_x - init_x, :end_y - init_y, :]
    return img1

def reconstruct_pred_map(probs, squares, img, k_stride, pred_shape):
    """
    Reconstruct the feature map from the predictions made
    on the squares representation
    Kernel_stride should divide image_size, yet usually the border which is
    lost is really thin
    -------------------
    pred_shape : size of the input by the network to make one prediction
    """
    pred_shape = ((img.shape[0] - pred_shape + 1) / k_stride,
                  (img.shape[1] - pred_shape + 1) / k_stride)
    pred_map = np.zeros(pred_shape)
    preds = np.reshape(probs[:, :, :, 0], squares.shape[0:2] + probs.shape[1:3])
    for i in xrange(preds.shape[0]):
        for j in xrange(preds.shape[1]):
            # Indices on the pred_map
            init_x = i * probs.shape[1]
            init_y = j * probs.shape[2]
            end_x = init_x + probs.shape[1]
            end_y = init_y + probs.shape[2]
            # Indices on preds[i,j], predictions over squares[i,j]
            end_i = preds[i, j].shape[0]
            end_j = preds[i, j].shape[1]

            if i == preds.shape[0] - 1:
                end_x = pred_shape[0]
                end_i = (end_x - init_x) / k_stride
            if j == preds.shape[1] - 1:
                end_y = pred_shape[1]
                end_j = (end_y - init_y) / k_stride

            pred_map[init_x: end_x, init_y: end_y] =\
                        preds[i, j, :end_i, :end_j]
    return pred_map

def square(image, size=None):
    """
    0-pad an image to get it square
    -------------------------------
    image : numpy array representing an image
    size : if precised, 0 pad to get image.shape = (size,size)
    """
    img = image.view()
    if size is None:
        size = max(img.shape)
    rval = np.zeros((size, size, 3), dtype='float32')
    rval[:img.shape[0], :img.shape[1], :] = img
    return rval

def testReconstruct():
    # Define image and model
    img_file = "/data/lisa/data/faces/FDDB/2002/08/11/big/img_276.jpg"
    model_file = '../../exp/2layers_16.pkl'
    scales = [0.8]
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

def get_max(arr_, n=20):
    """
    Return the indices of the n highest values in arr_
    """
    rval = []
    a = np.copy(arr_)
    for i in xrange(n):
        idx = np.argmax(a, axis=None)
        multi_idx = np.unravel_index(idx, a.shape)
        a[multi_idx] = 0.0
        rval.append(multi_idx)
    return rval


def fake_nms(res, n=20):
    for m in res:
        a = get_max(res[m],n)
        b = np.zeros(res[m].shape, dtype='float32')
        for idx in a:
            b[idx] = res[m][idx]
        res[m] = b
    return res


def get_rois(nms_maps, pred_size, pred_stride, enlarge_factor=0.0):
    """
    Return the coords of patches corresponding to
    non-zeros areas after nms execution
    rois : Regions of Interests
    ---------------------------------------------
    nms_maps : dict, maps of predictions indexed by scales
               versions obtained after performing NMS
    pred_size : size of the input patch of the classifier
    pred_stride : stride of classification

    RETURNS :
    rois : a list of 2*2 np_arrays indicating areas of interests
    """
    rois = []
    scores = []
    for s in nms_maps:
        n_z = np.transpose(np.nonzero(nms_maps[s]))
        for e in xrange(n_z.shape[0]):
            scores.append(nms_maps[s][n_z[e, 0], n_z[e,1]])
            # s defines a zoom
            stride = float(pred_stride) / float(s)
            size = float(pred_size) / float(s)
            enlarge_border = size * enlarge_factor / 2.0

            a = n_z[e, :] * stride - enlarge_border
            b = a + size + enlarge_border * 2

            rois.append(np.vstack((a, b)))
    return rois, scores


def rois_to_slices(rois):
    """
    Returns slices made from RoIs so that
    img[rval[i]] return the pixels corresponding to a RoI
    ------------------------------------------------------
    rois : list of 2*2 np arrays that may contain valid coords
           considering img_shape
    """
    rval = copy(rois)
    for i, e in enumerate(rval):
        rval[i] = [slice(e[0, 0], e[1, 0]),
                   slice(e[0, 1], e[1, 1]), slice(None)]
    return rval


def correct_rois(rois, img_shape):
    """
    Returns the cropped RoIs so that they are valid regions of the image
    -------------------------------------------------------------------
    rois : list of 2*2 numpy arrays
           usually result of enlarge can return regions reaching indices
           out of image shape, this corrects it
    img_shape : shape of the image, 3 int tuple
    """
    rval = copy(rois)
    for i, e in enumerate(rval):
        # Necessary to avoid copy effects !
        f = np.copy(e)
        f[0, :] = np.maximum(f[0, :], np.zeros((2,)))
        f[1, :] = np.minimum(f[1, :], np.array([img_shape[0:2]]))
        rval[i] = f
    return rval


def cascade(image_file, fprops, scales, sizes, strides, probs=None):
    """
    Perform the cascade classifier on image
    Returns the list of bounding boxes of the found items
    ---------------------------------------------
    image_file : path to an image
    fprops : list of fprop_funcs
                  will be used in the indexing order
    scales : list of lists of ints, define the scales used by each model
             will be used in the indexing order
    sizes : list of ints
            sizes of inputs for the corresponding classifiers
    strides : list of ints
              strides for the correspondiing classifiers
    probs : list of floats
            if not None, define acceptance prob for each classifier
    """

    assert len(fprops) == len(sizes)
    assert len(sizes) == len(strides)

    if probs is not None:
        assert len(strides) == len(probs)

    # We have to define scales according to the size of each predictor input
    base_size = float(max(sizes))  # will be 96 in general
    scales_ = []
    for i in xrange(len(fprops)):
        scales_.append([s * float(sizes[i]) / base_size for s in scales])

    # Perform first level
    img = cv2.imread(image_file)
    res = process_image(fprops[0], img, scales_[0])
######################################################
    res = fake_nms(res, n=5)
    #res = nms(sizes[0], strides[0], res)
######################################################
    rois, scores = get_rois(res, sizes[0], strides[0], enlarge_factor=0.3)
    rois = correct_rois(rois, img.shape)
    slices = rois_to_slices(rois)

    for i in xrange(1, len(fprops)):

        next_rois = []
        next_scores = []
        # For each RoI of the past level

        for j, sl in enumerate(slices):
            crop_ = img[sl]
            res_ = process_image(fprops[i], crop_, scales_[i])
######################################################
            #res_ = nms(sizes[i], strides[i], res_, probs[i])
            res_ = fake_nms(res_, n=1)
######################################################

            rois_, scores_ = get_rois(res_, sizes[i], strides[i],
                                      enlarge_factor=1)
            rois_ = correct_rois(rois_, crop_.shape)

            # Get the absolute coords of the new RoIs
            for r, s in zip(rois_, scores_):
                next_rois.append(r + rois[j][0, :])
                next_scores.append(s)
        rois = next_rois
        scores = next_scores
        # Get the slices from the absolute values
        slices = rois_to_slices(rois)
    return rois, scores


# # # TEST # # #

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


if __name__ == "__main__":
    # Define image and model
    img_file = "/data/lisa/data/faces/FDDB/2002/08/11/big/img_276.jpg"
    model_file1 = '../../exp/convtest/models/conv16_best.pkl'
    model_file2 = '../../exp/convtest/models/conv48_best.pkl'
    model_file3 = '../../exp/convtest/convTest96_best.pkl'

    img = cv2.imread(img_file)
    print img_file
    print model_file1
    print model_file2
    print img.shape, img.dtype

    with open(model_file1, 'r') as m_f:
        model1 = pkl.load(m_f)
    with open(model_file2, 'r') as m_f:
        model2 = pkl.load(m_f)
    with open(model_file3, 'r') as m_f:
        model3 = pkl.load(m_f)

    # Compile network prediction function
    x = T.tensor4("x")
    predict1 = function([x], model1.fprop(x))
    predict2 = function([x], model2.fprop(x))
    predict3 = function([x], model3.fprop(x))

    fprops = [predict1, predict2, predict3]
    sizes = [16, 48, 96]
    strides = [1, 1, 1]
    scales = [0.8]
    probs = [0.2, 0.2, 0.2]

    rois, scores = cascade(img_file, fprops, scales, sizes, strides, probs)
    print "Final RoIs :"
    for i in xrange(len(rois)):
        print rois[i], type(rois[i][0, 0])
        print scores[i]

    for i in xrange(len(rois)):
        cv2.rectangle(img, (int(rois[i][0, 0]), int(rois[i][0, 1])),
                      (int(rois[i][1, 0]), int(rois[i][1, 1])),
                      (0, 255, 0), 2)
        cv2.putText(img, str(i)+','+str(scores[i]),
                    (int(rois[i][0, 0]) + 10, int(rois[i][0, 1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0))

    cv2.imshow('r', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
