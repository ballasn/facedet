import numpy as np
from theano import function
import theano.tensor as T
import cPickle as pkl
import cv2
import sys
from utils.create_dataset.lists_to_hdf import image_from_line





if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: <text_file> <model> <out>" % sys.argv[0])
        sys.exit(1)

    # Parameters
    patch_size = 16
    nb_channels = 3
    acc_low = 0.1
    acc_high = 25
    text_file = sys.argv[1]
    out_file = sys.argv[3]
    model_file = sys.argv[2]
    batch_size = 128

    # Define the used model
    print 'Using model :', model_file
    with open(model_file, 'r') as f:
        model = pkl.load(f)
    x = T.tensor4('x')
    fp = function([x], model.fprop(x))


    f = open(text_file, 'r')
    lines = f.read().splitlines()
    f.close()

    tot = len(lines)
    with open(out_file, 'w') as f_out:
        mini_batch = np.zeros((batch_size, nb_channels, patch_size, patch_size),
                              dtype='float32')

        for i in xrange(len(lines)/batch_size):
            print i, len(lines)/batch_size

            for j in xrange(batch_size):
                k = i * batch_size + j

                patch = image_from_line(lines[k])
                patch = cv2.resize(patch, (patch_size, patch_size),
                                   interpolation=cv2.INTER_CUBIC)
                patch = np.transpose(patch, (2, 0, 1))
                patch =np.asarray(patch, dtype='float32')
                mini_batch[j, :] = patch

            #print mini_batch.shape
            scores = fp(mini_batch)
            #scores = fp(np.transpose(mini_batch, (1, 2, 3, 0)))
            #print scores.shape
            scores = scores[:, 0, :, :]
            #print scores.shape
            for e in range(len(scores)):
                if scores[e] > acc_low and scores[e] < acc_high:
                    f_out.write(lines[e+i*batch_size]+'\n')


    print 'Done'
