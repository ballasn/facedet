import numpy as np
from time import time

def process_in_chunks(fprop, examples, max_size=10**6):
    '''
    Process a list of examples in minibatches
    =========================================
    fprop: Theano function
        function to be applied on the minibatches
    examples : list
        examples to be processed, need to be the same size\
        Assumed to be images as 01C
    max_size :int
        nb of elements (eg float32) in each minibatch
        Decrease the value for old GPUs
    -----------------------------------
    rval : list
        list of the output in the same order as examples
    '''
    # size in number of elements
    ex = examples[0]
    size_example = ex.shape[0]*ex.shape[1]*ex.shape[2]

    # Create the batch of examples
    batch = np.zeros((len(examples), ex.shape[2], ex.shape[0], ex.shape[1]),
                     dtype='float32')
    for i in range(len(examples)):
        batch[i, :, :, :] = np.transpose(examples[i], (2, 0, 1))

    # Number of examples in a chunk
    chunk_size = max_size/size_example
    if chunk_size == 0:
        print 'max_size :', max_size, 'size of an example :', size_example
        print 'max_size is too low given the size of an example'
        exit(1)

    # Number of chunks to be produced
    chunk_nb = int(np.ceil(batch.shape[0]/float(chunk_size)))
    rval = None
    tt = 0
    for ch in range(chunk_nb):
        chunk = batch[ch*chunk_size:(ch+1)*chunk_size, :, :, :]
        if rval is None:
            rval = fprop(chunk)
        else:
            rval = np.concatenate((rval, fprop(chunk)))
    rval = [rval[i, 0, :, :] for i in range(rval.shape[0])]
    return rval


def thresholding(array, thresh, scale, slice_idx=0):
    '''
    Just get elements over thresh from a 2d np array
    '''
    array = array * (array > thresh)
    n_z = np.transpose(np.nonzero(array))
    rval = [[scale, n_z[e, 0], n_z[e, 1],
             array[n_z[e, 0], n_z[e, 1]], slice_idx]
            for e in range(len(n_z))]
    if rval != []:
        rval.sort(key=lambda x: x[3], reverse=True)
    return rval


def cut_in_numpy(img, k_stride, k_shape, max_side_size=700, nb_channels=3):
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
    nb_channels : number of channels in the batch
    ----------------------------------------------------
    OUTPUT :
    batch : np array containing the different pieces
    coords : coords of the top left pixel of each piece,
             indexed as batch
    """
    if max_side_size > img.shape[0]:
        x_pieces = 1
        x_size = img.shape[0]
        x_stride = 0
    else:
        x_size = max_side_size
        x_stride = max_side_size - (k_shape - k_stride)
        x_pieces = int(np.ceil((img.shape[0]-max_side_size) /
            float(x_stride)))+1

    if max_side_size > img.shape[1]:
        y_pieces = 1
        y_size = img.shape[1]
        y_stride = 0
    else:
        y_size = max_side_size
        y_stride = max_side_size - (k_shape - k_stride)
        y_pieces = int(np.ceil((img.shape[1]-max_side_size) /
            float(y_stride)))+1

    if y_pieces == 1 and x_pieces == 1:
        print 'Error, image should have been of larger size'
        exit(12)

    batch = np.zeros((x_pieces * y_pieces, nb_channels, max_side_size,
                     max_side_size), dtype='float32')
    coords = []
    cur = 0
    for i in xrange(x_pieces):
        for j in xrange(y_pieces):
            init_x = i * x_stride
            init_y = j * y_stride
            end_x = init_x + x_size
            end_y = init_y + y_size
            # We need square_size elements
            if i == x_pieces - 1:
                end_x = img.shape[0]
                init_x = min(end_x - max_side_size, init_x)
            if j == y_pieces - 1:
                end_y = img.shape[1]
                init_y = min(end_y - max_side_size, init_y)
            # Stored in BC01
            batch[cur, :, :, :] = \
                np.transpose(img[init_x: end_x, init_y: end_y, :], (2, 0, 1))
            coords.append([init_x, init_y])
            cur += 1

    return batch, coords
