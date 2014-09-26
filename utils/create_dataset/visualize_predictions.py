import numpy as np
from theano import function
import theano.tensor as T
import cPickle as pkl
from datasets.faceDataset_cascade import faceDataset
import matplotlib.pyplot as plt
import cv2
import sys
from time import time

def order_negative_response(predict, examples, batch_size=128,
                    nb_classes=2):
    """
    Return the sorted indices of negatives given their classification
        eg examples that have a highly negative response
    ----------------------------------------------
    model_file : pkl file containing the classifier
    dataset : dataset to be filtered
    acc_prob : threshold to consider an ex as inactive
            if p(face) < acc_prob then the ex is considered inactive
    """
    # Compute prediction of training set
    print "Defining inactives"
    res_ = []
    t0 = time()
    for n in xrange(len(examples)/batch_size):
        if n%20 == 0:
            t = time()
            sys.stdout.write('\r'+str(n)+', '+\
            str(t-t0)+' s for the last 20 minibatches')
            sys.stdout.flush()
            t0 = time()
        cur = n * batch_size
        end = (n+1) * batch_size
        x = examples[cur:end]
        # Transform into C01B format
        x = np.reshape(x, (len(x), 16, 16, 3))
        x = np.transpose(x, (3, 1, 2, 0))
        preds = predict(x)
        # Loop over examples
        for i in range(batch_size):
            res_.append([i + cur, preds[i, 0, 0, 0]])
        cur += batch_size
    res_.sort(key=lambda x: x[1])
    return res_


def getIndex(probs, p, start=0):
    '''
    Return the index of the first element higher than p
    '''
    for i, val in enumerate(probs[start:]):
        if val >= p:
            return i+start
    return len(probs)-1


if __name__ == '__main__':
    model_file = '../exp/convtest/models/large700_best.pkl'
    with open(model_file, 'r') as m_f:
        model = pkl.load(m_f)
    print len(model.layers)
    model.layers.pop()
    print len(model.layers)

    t0 = time()
    print 'Compiling fprop'
    x = T.tensor4('x')
    predict = function([x], model.fprop(x))
    t = time()
    print t-t0, 's to compile fprop'

    examples1 = "/data/lisatmp3/chassang/facedet/16/neg700_valid2.npy"
    examples2 = "/data/lisatmp3/chassang/facedet/16/pos700_valid2.npy"

# Classify examples
    examples1 = np.load(examples1)
    examples2 = np.load(examples2)

    examples = np.concatenate((examples1, examples2))
    ind_prob = order_negative_response(predict, examples)

    probs = [e[1] for e in ind_prob]
    inds = [e[0] for e in ind_prob]

    y = range(len(probs))

    print "nombre d'exemples :", len(probs)
    plt.plot(y, probs)
    plt.title('cumulative step')
    plt.show()
    print 'p(face) is given as title'
    i = 0
    print 'Up/Down to change to increase/decrease p(face)'
    print 'Left/Right to move along the examples'
    nb_display = 20
    per_row = 5

    while i < len(inds)/nb_display:
        for k in range(nb_display):
            j = i*nb_display + k
            img = np.reshape(examples[inds[j]], (16, 16, 3))
            img2 = np.reshape(examples[inds[-j]], (16, 16, 3))

            img = np.asarray(img, dtype='uint8')
            img2 = np.asarray(img2, dtype='uint8')

            title = str(round(probs[j], 4))+' '+str(j)
            title2 = str(round(probs[-j], 4))+' '+str(-j)
            print title, title2
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(title, img)
            img2 = cv2.resize(img2, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(title2, img2)

            cv2.moveWindow(title, 300+(k%per_row)*70, 300+(k/per_row)*90)
            cv2.moveWindow(title2, 800+(k%per_row)*70, 300+(k/per_row)*90)

        c = cv2.waitKey(0)
        cv2.destroyAllWindows()
        print 'code of the key :', c
        c = c % 256
        print 'invariant code  :', c
        # Define j according to the new key
        print 'i', i,
        print '   j', j
        if c == 83:  # Right arrow
            i += 1
        elif c == 81:  # Left arrow
            i -= 1
        elif c == 82:  # Up arrow
            print 'getting index for', probs[j] + 1.00, 'starting at', j
            print 'score at', j, ':', probs[j]
            i = getIndex(probs, probs[j] + 1.00, start=j)
            print 'got', i, 'with score', probs[i],
            i /= nb_display
            print 'transformed into', i

        elif c == 84:  # Down arrow
            print 'getting index for', probs[j] - 1.00, 'starting at', j
            print 'score at', j, ':', probs[j]
            i = getIndex(probs, probs[j] - 1.00, start=j) / nb_display
            print 'got', i, 'with score', probs[i],
            i /= nb_display
            print 'transformed into', i
        elif c == 27:  # Escape
            sys.exit(1)


