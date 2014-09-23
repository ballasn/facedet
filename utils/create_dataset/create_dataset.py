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


def create_positives_sample(img_list,
                            img_dir,
                            bbox_dir,
                            nb_ex,
                            bbox_ext='.txt',
                            nb_channels=3,
                            output_file='pos_list.txt'):

    # Initializing variables
    cur = 0
    chkpt_print = 0
    chkpt_save = 0
    nb_file = 0
    t1 = 0
    t2 = 0
    t0 = time()
    s = ''
    nb_draw = 5

    while cur < nb_ex:
        # print cur, img, nb_box
        if cur % 100 == 0 and cur != 0 and cur != chkpt_print:
            chkpt_print = cur
            t1 = time()
            sys.stdout.write("\r"+str(cur)+", last 100 in "+str(t1-t2)+" s" +
                             ", total : "+str(nb_file)+" files or " +
                             str(float(t1-t0)/float(chkpt_print))+" s/image")
            sys.stdout.flush()
            t2 = time()

        # Saving frequently
        if cur % 10000 == 0 and cur != chkpt_save:
            with open(output_file, 'a') as output:
                output.write(s)
            s = ''
            chkpt_save = cur
            print "temp save :", cur, "examples"
            print ""

        # Pick a random image
        idx = np.random.randint(0, len(img_list))
        img = img_list[idx]

        # Get image filename
        filepath = os.path.join(img_dir, img).strip()
        bbox_file = os.path.join(bbox_dir, os.path.splitext(img)[0] + bbox_ext)

        # Pursue only if the image exists and has faces
        if (not os.path.isfile(filepath) or not os.path.isfile(bbox_file)):
            continue

        # Check image is OK
        img = cv2.imread(filepath)
        if img is None:
            print "Loading failed: [%s]" % filepath
            exit(1)

        # Get the boxes
        bbox = np.loadtxt(skipper(bbox_file), delimiter=',', dtype=int)
        if len(bbox.shape) == 1:
            bbox = np.reshape(bbox, (1, bbox.shape[0]))

        for b in xrange(0, bbox.shape[0]):

            box = bbox[b, 2:]  # x, y, w, h
            for j in range(nb_draw):

                size = int(0.9 * min(box[2], box[3]))
                x0 = np.random.randint(0, int(0.1 * box[2])+1)
                y0 = np.random.randint(0, int(0.1 * box[3])+1)

                # Save the patch
                s += img_list[idx][:-1]+' '+str(box[0]+x0)+' ' +\
                    str(box[1]+y0)+' '+str(box[0]+x0+size)+' ' +\
                    str(box[1]+y0+size)+'\n'

                # Save the patch
                cur += 1

    with open(output_file, 'a') as output:
        output.write(s)

    return s


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


def range_overlap(a_min, a_max, b_min, b_max):
    '''
    Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)


def bbox_overlap(b1, b2):
    '''
    Overlapping rectangles overlap both horizontally & vertically
    '''
    h_overlap = range_overlap(b1[0], b1[2], b2[0], b2[2])
    v_overlap = range_overlap(b1[1], b1[3], b2[1], b2[3])
    return h_overlap and v_overlap


def score_iou(b1, b2):
    if not bbox_overlap(b1, b2):
        return 0.0
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    inter = (x1-x0) * (y1-y0)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    den = min(area1, area2)
    score = float(inter)/float(den)
    return score


def score_iomin(b1, b2):
    if not bbox_overlap(b1, b2):
        return 0.0
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    inter = (x1-x0) * (y1-y0)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    den = min(area1, area2)
    score = float(inter)/float(den)
    return score


def create_negative_sample(img_list,
                           nb_samples,
                           img_dir,
                           bbox_dir,
                           path,
                           bbox_ext='.txt',
                           nb_channels=3,
                           predict=None,
                           max_overlap=0.2,
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
        if cur % 100 == 0 and cur != 0 and cur != chkpt_print:
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
        img = img_list[idx]

        # Get image filename
        filepath = os.path.join(img_dir, img).strip()
        bbox_file = os.path.join(bbox_dir, os.path.splitext(img)[0] + bbox_ext)

        # Pursue only if the image has a bounding box
        if (not os.path.isfile(bbox_file)):
            continue

        # Load data of the image
        img = cv2.imread(filepath)
        nb_file += 1
        if img is None:
            print "Loading failed: [%s]" % filepath
            exit(1)

        # Load face areas
        bbox = np.loadtxt(skipper(bbox_file), delimiter=',', dtype=int)
        if len(bbox.shape) == 1:
            bbox = np.reshape(bbox, (1, bbox.shape[0]))
        bbox = bbox[:, 2:]

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

            # Check for overlaps among faces
            for b in xrange(bbox.shape[0]):
                box = np.copy(bbox[b, :])
                box[2:] += box[:2]

                # Inter over Union
                score = score_iomin(sample_box, box)
                ok = ok and (score < max_overlap)

                # Display for test purpose
                """
                cv2.rectangle(img,
                              (box[1], box[0]),
                              (box[3], box[2]),
                              (0, 255, 0), 3)
                cv2.putText(img, str(b)+', '+str(score),
                            (box[1] + 15, box[0] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0))
                """
            # We want negatives that look like positives
            if predict is not None:
                ex = cv2.resize(pixel_box, (16, 16),
                                interpolation=cv2.INTER_CUBIC)
                ex = np.reshape(ex, (1,) + ex.shape)
                ex = np.transpose(ex, (3, 1, 2, 0))

                prob = predict(ex)

                ok = ok and prob[0, 0, 0, 0] >= acc[0]\
                        and prob[0, 0, 0, 0] < acc[1]

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

def positivesFromDataset(googledataset):
    """
    Returns a nparray containing the images of faces
    The argument is GoogleDataset object
    """
    for i in xrange(len(googledataset)):
        # Returns a facedatasetExample
        img = googledataset[i]


def load_list(filename):
    id_list = []
    with open(filename, 'r') as fd:
        for line in fd:
            id_list.append(line)
    return id_list


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print("Usage %s: <img_dir><bbox_dir> <img_list> <negative> <out_file>"
              % sys.argv[0])
        exit(1)

    img_dir = sys.argv[1]
    bbox_dir = sys.argv[2]
    img_files = load_list(sys.argv[3])
    negative = int(sys.argv[4])
    out_file = sys.argv[5]

    if (negative > 0):
        print "Creating a list of", negative, "negatives"
        # Define the used model
        model_file = '../../exp/convtest/models/curiculum2_best.pkl'
        print 'Using model :', model_file
        with open(model_file, 'r') as f:
            model = pkl.load(f)
        x = T.tensor4('x')
        fp = function([x], model.fprop(x))

        t0 = time()
        create_negative_sample(img_files, negative,
                               img_dir, bbox_dir, sys.argv[6],
                               predict=fp,
                               output_file=out_file)
        t = time()
        print t-t0, 's for', negative, 'examples'

    else:
        print "Creating a list of", -negative, "positives"
        t0 = time()
        create_positives_sample(img_files, img_dir,
                                bbox_dir, -negative,
                                output_file=out_file)
        t = time()
        print t-t0, 's for', -negative, 'examples'
