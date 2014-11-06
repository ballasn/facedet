import numpy as np
import cv2
import sys


def process_image(fprop_func, image, scales, pred_stride, pred_size, piece_size=700):
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

    map_ = apply_fprop(fprop_func, minibatch, pred_stride, pred_size, piece_size)

    return map_


def apply_fprop(fprop_func, image, k_stride, k_shape, piece_size=700):
    """
    Apply a model on a image, returns the map of p(face)
    ------------------------
    fprop_func : a compiled fprop function
    image : a numpy array representing an image
    """
    if type(image) == dict:
        rval = {}
        for s in image:
            rval[s] = apply_fprop(fprop_func, image[s], k_stride, k_shape,
                                  piece_size)
        return rval

    # Add a minibatch dim to get C01B format
    # print 'before reshape', image.shape,
    img_ = np.reshape(image, list(image.shape)+[1])
    # print 'after reshape', image.shape,
    img_ = np.transpose(img_, (2, 0, 1, 3))
    # print 'after transpose', image.shape,
    img_ = np.transpose(img_, (3, 0, 1, 2))

    # Need to deal with Out of memory error
    try:
        rval = fprop_func(img_)
    except RuntimeError:  # Because sometimes it isn't a memory error
        print sys.exc_info()[1]
        print 'RunTime Error'
        print 'Usually happens in this case because the image is too large.',
        print 'Try a lower piece_size'
        sys.exit(42)

    # Here we need to reconstruct the output if in pieces
    # Sigmoid version : 01BP
    # Softmax now keeps the format
    # If BC01 used, the output is BP01

    # BP01 : Softmax
    return rval[0, 0, :, :]
    # 01BP : Sigmoid
    return rval[:, :, 0, 0]


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


def cut_in_pieces(img, k_stride, k_shape, max_side_size=700):
    """
    Cut an image into pieces to be classified independently
    Enables to process large images without mem overload
    The overlap between pieces is k_shape - k_stride, eg
    we mimic taking the whole image as input
    ----------------------------------------------------
    img : numpy array representing an image
    k_stride : stride of the kernel
    k_shape : shape of the kernel
    square_size = size of the extracted images
    """
    if max_side_size > img.shape[0]:
        x_pieces = 1
        x_size = img.shape[0]
        x_stride = 0
    else:
        x_size = max_side_size
        x_stride = max_side_size - (k_shape - k_stride)
        x_pieces = int(img.shape[0] / x_stride + 1)

    if max_side_size > img.shape[1]:
        y_pieces = 1
        y_size = img.shape[1]
        y_stride = 0
    else:
        y_size = max_side_size
        y_stride = max_side_size - (k_shape - k_stride)
        y_pieces = int(img.shape[1] / y_stride + 1)

    if y_pieces == 1 and x_pieces == 1:
        return img

    pieces = np.zeros((x_pieces, y_pieces, x_size, y_size, 3),
                      dtype='float32')
    for i in xrange(x_pieces):
        for j in xrange(y_pieces):
            init_x = i * x_stride
            init_y = j * y_stride
            # We need square_size elements
            end_x = init_x + x_size
            end_y = init_y + y_size
            if i == x_pieces - 1:
                end_x = img.shape[0]
                init_x = end_x - x_size
            if j == y_pieces - 1:
                end_y = img.shape[1]
                init_y = end_y - y_size
            pieces[i, j, :, :, :] =\
                    img[init_x: end_x, init_y: end_y, :]
    return pieces

def cut_in_dict(img, k_stride, k_shape, max_side_size=700):
    """
    Cut an image into pieces to be classified independently
    Enables to process large images without mem overload
    The overlap between pieces is k_shape - k_stride, eg
    we mimic taking the whole image as input
    ----------------------------------------------------
    img : numpy array representing an image
    k_stride : stride of the kernel
    k_shape : shape of the kernel
    square_size = size of the extracted images
    """
    if max_side_size > img.shape[0]:
        x_pieces = 1
        x_size = img.shape[0]
        x_stride = 0
    else:
        x_size = max_side_size
        x_stride = max_side_size - (k_shape - k_stride)
        x_pieces = int(np.ceil(img.shape[0] / float(x_stride)))

    if max_side_size > img.shape[1]:
        y_pieces = 1
        y_size = img.shape[1]
        y_stride = 0
    else:
        y_size = max_side_size
        y_stride = max_side_size - (k_shape - k_stride)
        y_pieces = int(np.ceil(img.shape[1] / float(y_stride)))

    if y_pieces == 1 and x_pieces == 1:
        print 'Error, image should have been of larger size'
        exit(12)
    # pieces will be a dict indexed by top left pixels of pieces
    # This enables different sizes of pieces
    pieces = {}
    for i in xrange(x_pieces):
        for j in xrange(y_pieces):
            init_x = i * x_stride
            init_y = j * y_stride
            # We need square_size elements
            end_x = init_x + x_size
            end_y = init_y + y_size
            if i == x_pieces - 1:
                end_x = img.shape[0]
                init_x = min(end_x - k_shape, init_x)
            if j == y_pieces - 1:
                end_y = img.shape[1]
                init_y = min(end_y - k_shape, init_y)
            pieces[(init_x, init_y)] = img[init_x: end_x, init_y: end_y, :]
    return pieces


def get_init(i, j, img_shape, k_stride, k_shape, pieces, piece_size=700):
    """
    Returns the coords of the top-left pixel of pieces[i,j]
    """
    x_stride = min(piece_size, img_shape[0]) - (k_shape - k_stride)
    y_stride = min(piece_size, img_shape[1]) - (k_shape - k_stride)
    init_x = i * x_stride
    init_y = j * y_stride
    if i == pieces.shape[0] - 1:
        init_x = img_shape[0] - pieces.shape[2]
    if j == pieces.shape[1] - 1:
        init_y = img_shape[1] - pieces.shape[3]
    return init_x, init_y


def reconstruct(pieces, img_shape, k_stride, k_shape, piece_size=700):
    """
    Rebuilds the image from the pieces
    Useful for test purpose, you can check that the pieces
    were created the right way
    """
    img1 = np.zeros(img_shape)
    for i in xrange(pieces.shape[0]):
        for j in xrange(pieces.shape[1]):
            init_x, init_y = get_init(i, j, img_shape,
                                      k_stride, k_shape, pieces, piece_size)
            end_x = init_x + pieces.shape[2]
            end_y = init_y + pieces.shape[3]
            if i == pieces.shape[0] - 1:
                end_x = img_shape[0]
                init_x = end_x - pieces.shape[2]
            if j == pieces.shape[1] - 1:
                end_y = img_shape[1]
                init_y = end_y - pieces.shape[3]

            img1[init_x: end_x, init_y: end_y, :] =\
                    pieces[i, j, :, :, :]
    return img1


def reconstruct_pred_map(probs, pieces, img, k_stride, pred_shape):
    """
    Reconstruct the feature map from the predictions made
    on the pieces representation
    Kernel_stride should divide image_size, yet usually the border which is
    lost is really thin
    -------------------
    pred_shape : size of the input by the network to make one prediction
    """
    pred_shape = ((img.shape[0] - pred_shape + 1) / k_stride,
                  (img.shape[1] - pred_shape + 1) / k_stride)
    pred_map = np.zeros(pred_shape)
    preds = np.reshape(probs[:, :, :, 0],
                       pieces.shape[0:2] + probs.shape[1:3])
    for i in xrange(preds.shape[0]):
        for j in xrange(preds.shape[1]):
            # Indices on the pred_map
            init_x = i * probs.shape[1]
            init_y = j * probs.shape[2]
            end_x = init_x + probs.shape[1]
            end_y = init_y + probs.shape[2]
            # Indices on preds[i,j], predictions over pieces[i,j]
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

if __name__ == '__main__':
    print '*'*50
    k_stride = 1
    k_shape = 1
    a = 10000 * np.random.rand(3, 3, 3)
    print a.shape
    p = cut_in_pieces(a, k_stride, k_shape, max_side_size=2)
    print p.shape
    b = reconstruct(p, a.shape, k_stride, k_shape, 2)
    print b.shape
    print a.dtype, b.dtype
    print a-b
    print np.allclose(a, b)
    print '*'*50


