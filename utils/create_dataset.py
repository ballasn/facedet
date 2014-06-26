import sys
import os
import cv2
import numpy as np

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
    for img in img_list:
        bbox_file = os.path.join(bbox_dir, os.path.splitext(img)[0] + bbox_ext)
        if (os.path.isfile(bbox_file)):
            # print "Load:", bbox_file
            bbox = np.loadtxt(skipper(bbox_file), delimiter=',')
            nb_box += bbox.shape[0]


    print nb_box
    ### Initialize resulting matrix
    data = np.zeros((100000, patch_size[0] * patch_size[1] * nb_channels), dtype=np.float32)
    cur = 0

    for img in img_list:
        print cur, img, nb_box
        if cur > 100000:
            break

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
            print bbox[b, :]
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
    if (not size in [48, 96]):
        print "Invalid patch size", size
        exit(1)
    patch_size= (size, size)

    # if (negative > 0):
    #     mat = create_negatives_samples(l, negative, patch_size,
    #                                    img_dir, bbox_dir)
    # else:
    data = create_positives_sample(img_files, patch_size,
                                   img_dir, bbox_dir)

    np.save(sys.argv[6], data)
