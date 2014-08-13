import sys
import os
import cv2
import numpy as np
from time import time

RED = (0, 0, 255)

def skipper(fname):
    with open(fname) as fin:
        no_comments = (line for line in fin if not line.lstrip().startswith('#'))
        next(no_comments, None) # skip header
        for row in no_comments:
            yield row

def create_positives_sample(img_list,
                            patch_size,
                            img_dir,
                            bbox_dir,
                            bbox_ext = '.txt',
                            nb_channels = 3):

    # ### Get the number of bounding box
    nb_box = 0
    """
    for img in img_list:
        if nb_box%100 == 0:
            sys.stdout.write("\r"+str(nb_box)+" bboxes")
            sys.stdout.flush()
        bbox_file = os.path.join(bbox_dir, os.path.splitext(img)[0] + bbox_ext)
        if (os.path.isfile(bbox_file)):
            # print "Load:", bbox_file
            bbox = np.loadtxt(skipper(bbox_file), delimiter=',')
            nb_box += bbox.shape[0]
    """

    nb_box = 700000
    ### Initialize resulting matrix
    data = np.zeros((nb_box, patch_size[0] * patch_size[1] * nb_channels), dtype=np.float32)
    cur = 0

    for img in img_list:
        #print cur, img, nb_box
        if cur >= len(data):
            break
        if cur%100 == 0:
            sys.stdout.write("\r"+str(cur))
            sys.stdout.flush()

        ### Get image filenames
        filepath = os.path.join(img_dir, img).strip()
        bbox_file = os.path.join(bbox_dir, os.path.splitext(img)[0] + bbox_ext)

        ### Pursue only if the image has a bounding box
        if (not os.path.isfile(bbox_file)):
            continue

        ### Load data
        img = cv2.imread(filepath)
        if img == None:
            print "Loading failed: [%s]" % filepath
            exit (1)
        bbox = np.loadtxt(skipper(bbox_file), delimiter = ',', dtype=int)
        if len(bbox.shape) == 1:
            bbox = np.reshape(bbox, (1, bbox.shape[0]))

        for b in xrange(0, bbox.shape[0]):
            #print bbox[b, :]
            ### Debug show rectange
            # cv2.rectangle(img,
            #               (bbox[b, 3], bbox[b, 2]),
            #               (bbox[b, 3] + bbox[b, 5], bbox[b, 2] + bbox[b, 4]),
            #               RED, 3)

            face = img[bbox[b, 2]:bbox[b, 2] + bbox[b, 4], bbox[b, 3]:bbox[b, 3] + bbox[b, 5]]
            face = cv2.resize(face, patch_size, interpolation = cv2.INTER_CUBIC)

            ### Debug show images
            # cv2.imshow('image', face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            ### Save the patch
            data[cur, :] = face.reshape(1, patch_size[0] * patch_size[1] * nb_channels)
            cur += 1

    return data



def rnd_bounding_box(nrows, ncols):

    ### Get initial Position
    row = min(nrows - 10, np.random.random_integers(0, int(np.ceil(0.7 * nrows))))
    col = min(ncols - 10, np.random.random_integers(0, int(np.ceil(0.7 * ncols))))

    ### Get Size
    bbox_nrows = max(nrows - row - 5, int(np.ceil(np.random.rand() * 0.3 * nrows)))
    bbox_ncols = max(nrows - row - 5, int(np.ceil(np.random.rand() * 0.3 * ncols)))

    return np.array([[0, 0, row, col, bbox_nrows, bbox_ncols]])


def range_overlap(a_min, a_max, b_min, b_max):
    '''
    Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def bbox_overlap(b1, b2):
    '''
    Overlapping rectangles overlap both horizontally & vertically
    '''

    h_overlap = range_overlap(b1[2], b1[2] + b1[4], b2[2], b2[2] + b2[4])
    v_overlap = range_overlap(b1[3], b1[3] + b1[5], b2[3], b2[3] + b2[5])
    return h_overlap and v_overlap



def create_negative_sample(img_list,
                           nb_samples,
                           patch_size,
                           img_dir,
                           bbox_dir,
                           path,
                           bbox_ext = '.txt',
                           nb_channels = 3):

    nb_draw = 100

    # Initialize resulting matrix
    data = np.zeros((nb_samples, patch_size[0] * patch_size[1] * nb_channels), dtype=np.float32)

    cur = 0
    chkpt_print = 0
    chkpt_save = 0
    nb_file = 0
    t1 = 0
    t2 = 0
    t0 = time()
    while cur < nb_samples:
        if cur % 100 == 0 and cur != 0 and cur != chkpt_print:
            chkpt_print = cur
            t1 = time()
            sys.stdout.write("\r"+str(cur)+", last 100 in "+str(t1-t2)+" s"+
            ", total : "+ str(nb_file) +" files or "+
            str(float(t1-t0)/float(chkpt_print))+" s/image")
            sys.stdout.flush()
            t2 = time()

        if cur % 10000 == 0 and cur != chkpt_save:
            np.save(path, data)
            chkpt_save = cur
            print "temp save :", cur, "examples"
            print ""

        idx = np.random.randint(0, len(img_list))
        img = img_list[idx]
        # print cur, img, nb_samples

        # Get image filenames
        filepath = os.path.join(img_dir, img).strip()
        bbox_file = os.path.join(bbox_dir, os.path.splitext(img)[0] + bbox_ext)

        # Pursue only if the image has a bounding box
        if (not os.path.isfile(bbox_file)):
            continue

        # Load data
        img = cv2.imread(filepath)
        nb_file += 1
        if img is None:
            print "Loading failed: [%s]" % filepath
            exit(1)
        # Load face areas
        bbox = np.loadtxt(skipper(bbox_file), delimiter=',', dtype=int)
        if len(bbox.shape) == 1:
            bbox = np.reshape(bbox, (1, bbox.shape[0]))

        for i in xrange(0, nb_draw):
            # Draw a random box
            sample_box = rnd_bounding_box(img.shape[0], img.shape[1])

            overlap = False
            for b in xrange(0, bbox.shape[0]):
                # Check if isn't overlapping a face
                if bbox_overlap(sample_box[0, :], bbox[b, :]):
                    overlap = True
                    break
            # Draw a new box
            if overlap:
                continue



            ### Debug show rectange
            # print img.shape, sample_box
            # cv2.rectangle(img,
            #               (sample_box[0, 3], sample_box[0, 2]),
            #               (sample_box[0, 3] + sample_box[0, 5],
            #                sample_box[0, 2] + sample_box[0, 4]),
            #               RED, 3)
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # Define the data as pixels inside the valid_box
            valid_box = img[sample_box[0, 2]:sample_box[0, 2] + sample_box[0, 4], sample_box[0, 3]:sample_box[0, 3] + sample_box[0, 5]]
            # Resize box to fit <patch_size>
            valid_box = cv2.resize(valid_box, patch_size, interpolation =
                    cv2.INTER_LINEAR)

            ### Debug show images
            # cv2.imshow('image', face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            ### Save the patch
            #print cur, data.shape
            data[cur, :] = valid_box.reshape(1, patch_size[0] * patch_size[1] * nb_channels)
            cur += 1
            if cur >= nb_samples:
                break

    return data

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

    if len(sys.argv) != 7:
        print("Usage %s: img_dir bbox_dir img_list negative size out" % sys.argv[0])
        exit(1)

    img_dir = sys.argv[1]
    bbox_dir = sys.argv[2]

    img_files = load_list(sys.argv[3])
    negative = int(sys.argv[4])
    size = int(sys.argv[5])
    if (not size in [16, 48, 96]):
        print "Invalid patch size", size
        exit(1)
    patch_size= (size, size)

## Counting bboxes
    """
    nb_box = 0
    bbox_ext = ".txt"
    for img in img_files:
        if nb_box%100 == 0:
            sys.stdout.write("\r"+str(nb_box)+" bboxes")
            sys.stdout.flush()
        bbox_file = os.path.join(bbox_dir, os.path.splitext(img)[0] + bbox_ext)
        print bbox_file
        if (os.path.isfile(bbox_file)):
            # print "Load:", bbox_file
            bbox = np.loadtxt(skipper(bbox_file), delimiter=',')
            nb_box += bbox.shape[0]
    print nb_box
    """
####
    if (negative > 0):
        print "Negatif"
        data = create_negative_sample(img_files, negative, patch_size,
                                     img_dir, bbox_dir, sys.argv[6])
    else:
        print "Positif"
        data = create_positives_sample(img_files, patch_size,
                                       img_dir, bbox_dir)

    np.save(sys.argv[6], data)
