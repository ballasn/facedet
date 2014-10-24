import sys
import os
import cv2
import numpy as np
from time import time
import numpy as np
from random import shuffle


def image_from_line(line, size,
                    data_dir='/data/lisa/data/faces/GoogleEmotionDataset'):
    """
    Return the corresponding patch from a line of text
    --------------------------------------------------
    line : string defining a patch as
           file x0 y0 x1 y1
    """
    row = line.split(' ')
    filepath = os.path.join(data_dir, row[0])
    img = cv2.imread(filepath)
    #print row, img.shape
    #cv2.imshow('Img', img)
    box = [int(e) for e in row[1:]]
    patch = rescale_size(box, size, img)

    #patch = img[int(max(0, int(row[2]))):int(min(img.shape[0], int(row[4]))),
    #            int(max(0, int(row[1]))):int(min(img.shape[1], int(row[3]))), :]

    ### Warning need to be inverted for FaceDataset
    #patch = img[int(max(0, int(row[1]))):int(min(img.shape[0], int(row[3]))),
    #            int(max(0, int(row[2]))):int(min(img.shape[1], int(row[4]))), :]
    #patch = np.asarray(patch, dtype='float32')
    return patch


def rescale_size(box, size, img):
    """
    box : [x, y, w, h]
    size : target size
    img : source from which we extract
    """
    if box[2] > box[3]:
        m = 3
        M = 2
    else:
        m = 2
        M = 3

    # Resize the longest axis
    ratio = 16.0 / box[M]
    new_l = int(ratio * box[m])
    # Number of pixels of the original image missing to have a square
    dl = (16 - new_l) / ratio
    r = dl/2
    new_box = list(box)
    #print 'm :', m,
    #print 'M :', M

    if box[m%2] - dl/2 < 0:
        r = dl - box[m%2]
        new_box[m%2] = 0
        new_box[m] += r
    elif box[m%2] + box[m] + dl/2 > img.shape[m%2]:
        r = dl - img.shape[m%2] + box[m%2] + box[m]
        new_box[m] += dl
        new_box[m%2] = img.shape[m%2] - box[m]
    #print 'old_box', box
    #print 'new_box', new_box
    #print img.shape
    patch = cv2.resize(img[new_box[0]:new_box[0]+new_box[2],
                           new_box[1]:new_box[1]+new_box[3]],
                       (size, size), interpolation=cv2.INTER_CUBIC)
    return patch

def npy_from_textfile(text_file, patch_size, output,
                      nb_channels=3,
                      data_dir='/data/lisa/data/faces/GoogleEmotionDataset'):

    # get list of patches
    t_file = open(text_file, 'r')
    lines = t_file.read().splitlines()  # remove \n at EOL
    t_file.close()

    thres = 700000

    nb_patches = min(len(lines), thres)

    # Create hdf output
    out_shape = (nb_patches, patch_size * patch_size * nb_channels)
    out = np.zeros(out_shape, dtype='float32')


    t0 = time()
    for cur, line in enumerate(lines):

        sys.stdout.write('\r'+str(cur)+' / '+str(nb_patches)+' '+line)
        sys.stdout.flush()
        if (cur >= nb_patches):
            break

        patch = image_from_line(line, patch_size, data_dir)
        #cv2.imshow('patch', patch)
        #patch = cv2.resize(patch, (patch_size, patch_size),
        #                   interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('patchresize', patch)
        #cv2.waitKey(0)
        patch = patch.reshape(1, patch_size * patch_size * nb_channels)

        out[cur, :] = patch

    np.save(output, out)


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: <text_file> <patch_size> <output_file>" % sys.argv[0])
        sys.exit(1)

    text_file = sys.argv[1]
    patch_size = int(sys.argv[2])
    output_file = sys.argv[3]
    print patch_size

    npy_from_textfile(text_file, patch_size, output_file)







