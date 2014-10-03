import numpy as np
import cv2


def process_image(fprop_func, image, scales, pred_size):
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
        img_ = rescale(image, s, pred_size)
        minibatch[s] = img_

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
    #print 'before reshape', image.shape,
    image = np.reshape(image, list(image.shape)+[1])
    #print 'after reshape', image.shape,
    image = np.transpose(image, (2, 0, 1, 3))
    #print 'after transpose', image.shape,
    image = np.transpose(image, (3, 0, 1, 2))
    rval = fprop_func(image)

    #rval = np.transpose(rval, (0, 2, 3, 1))
    #print 'predict size', rval.shape
    # Softmax now keeps the format
    # If BC01 used, the output is BP01
    # prob_map = rval[0, 0, :, :]
    #print 'shape of probs map', prob_map.shape
    return rval[0, 0, :, :]


def rescale(image, scale, pred_size):
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
                               (max(int(sh[1] * scale), pred_size),
                                max(int(sh[0] * scale), pred_size)),

                               interpolation=cv2.INTER_CUBIC)

    resized_array = np.asarray(resized_image, dtype=image.dtype)

    return resized_array


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
            squares[i, j, :end_x - init_x, :end_y - init_y, :] =\
                    img[init_x: end_x, init_y: end_y, :]

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
    preds = np.reshape(probs[:, :, :, 0],
                       squares.shape[0:2] + probs.shape[1:3])
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
