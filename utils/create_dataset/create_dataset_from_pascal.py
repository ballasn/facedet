import sys
import os
import cv2
import numpy as np
from time import time
import theano.tensor as T
from theano import function
import cPickle as pkl

RED = (0, 0, 255)


def skipper(fname):
    with open(fname) as fin:
        no_comments = (line
                       for line in fin if not line.lstrip().startswith('#'))
        next(no_comments, None)  # skip header
        for row in no_comments:
            yield row


def rnd_bounding_box(nrows, ncols, size=None):
    if size is None:
        # Get initial Position
        x0 = np.random.random_integers(0, int(np.ceil(0.6 * nrows)))
        y0 = np.random.random_integers(0, int(np.ceil(0.6 * ncols)))

        side = np.random.random_integers(0.2*min(nrows, ncols),
                                         0.4*min(nrows, ncols))
    else:
        x0 = np.random.random_integers(0, nrows - size)
        y0 = np.random.random_integers(0, ncols - size)
        side = size

    return np.array([x0, y0, x0+side, y0+side])


def create_negative_sample(img_list,
                           nb_samples,
                           nb_channels=3,
                           predict=None,
                           acc=[0.2, 0.8],
                           output_file='./dataset.txt'):

    # Initialize variables
    cur = 0
    chkpt_print = 0
    chkpt_save = 0
    nb_file = 0
    t1 = 0
    t2 = 0
    t0 = time()
    s = ''
    nb_draw = 10

    # Loop to create all the samples
    while cur < nb_samples:
        # Printing information about the flow

        if cur % 10 == 0 and cur != 0 and cur != chkpt_print:
            chkpt_print = cur
            t1 = time()
            sys.stdout.write("\r"+str(cur)+", last 100 in "+str(t1-t2)+" s" +
                             ", total : "+str(nb_file)+" files or " +
                             str(float(t1-t0)/float(chkpt_print))+" s/image")
            sys.stdout.flush()
            t2 = time()

        # Saving frequently
        if cur % 10000 == 0 and cur != chkpt_save:
            with open(out_file, 'a') as output:
                output.write(s)
            s = ''
            chkpt_save = cur
            print "temp save :", cur, "examples"
            print ""

        # Pick a random image
        idx = np.random.randint(0, len(img_list))
        filename = img_list[idx].strip()

        # Load data of the image
        img = cv2.imread(filename)
        nb_file += 1
        if img is None:
            print "Loading failed: [%s]" % filename
            exit(1)

        for i in xrange(0, nb_draw):

            # for display
            #img = cv2.imread(filepath)

            # Draw one random box
            sample_box = rnd_bounding_box(img.shape[0], img.shape[1])
            pixel_box = img[sample_box[0]:sample_box[2],
                            sample_box[1]:sample_box[3]]

            # Remove images too dark
            if pixel_box.mean() < 30:
                continue
            ok = True

            # We want negatives that look like positives
            if predict is not None:
                ex = cv2.resize(pixel_box, (16, 16),
                                interpolation=cv2.INTER_CUBIC)
                ex = np.reshape(ex, (1,) + ex.shape)
                ex = np.transpose(ex, (3, 1, 2, 0))
                prob = predict(ex)
                ok = ok and prob[0, 0, 0, 0] >= acc[0]

                # Display for test purpose
                """
                prob_ = round(float(prob[0, 0, 0, 0]), 8)

                cv2.putText(img, 'p(face) : '+str(prob_),
                            (box[1] + 15, box[0] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0))

            # Draw the random box
            cv2.rectangle(img,
                          (sample_box[1], sample_box[0]),
                          (sample_box[3], sample_box[2]),
                          RED, 3)


            if not ok:
                continue
            cv2.imshow('Original image'+str(i), img)
            cv2.moveWindow('Original image', 200, 200)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            if not ok:
                continue

            # cv2.imshow('img', pixel_box)
            # ex = cv2.resize(pixel_box, (16, 16),
            #                 interpolation=cv2.INTER_CUBIC)
            # cv2.imshow('img2', ex)
            # cv2.waitKey(0)


            # Save the patch
            s += img_list[idx][:-1]+' '+str(sample_box[0])+' ' +\
                str(sample_box[1])+' '+str(sample_box[2])+' ' +\
                str(sample_box[3])+'\n'

            cur += 1
            if cur >= nb_samples:
                break

    with open(out_file, 'a') as output:
        output.write(s)
    return s


def load_list(filename):
    id_list = []
    with open(filename, 'r') as fd:
        for line in fd:
            id_list.append(line)
    return id_list


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: <img_list> <nb_img> <out>" % sys.argv[0])
        exit(1)

    img_list = load_list(sys.argv[1])
    nb_samples = int(sys.argv[2])
    out_file = sys.argv[3]

    print "Creating a list of", nb_samples, "negatives"
    # Define the used model
    model_file = '/data/lisatmp3/chassang/facedet/models/16/curiculum2_best.pkl'
    print 'Using model :', model_file
    with open(model_file, 'r') as f:
        model = pkl.load(f)
        x = T.tensor4('x')
        fp = function([x], model.fprop(x))
        t0 = time()
        create_negative_sample(img_list,
                               nb_samples,
                               predict=fp,
                               output_file=out_file)
        t = time()
        print t-t0, 's for', nb_samples, 'examples'
