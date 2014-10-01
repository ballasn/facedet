import sys
import os
import cv2
import numpy as np
from time import time
import numpy as np
from random import shuffle


def image_from_line(line,
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
    #print max(0, row[2]), min(img.shape[0], row[4])
    #print max(0, row[1]), min(img.shape[1], row[3])
    patch = img[int(max(0, int(row[2]))):int(min(img.shape[0], int(row[4]))),
                int(max(0, int(row[1]))):int(min(img.shape[1], int(row[3]))), :]

    ### Warning need to be inverted for FaceDataset
    patch = img[int(max(0, int(row[1]))):int(min(img.shape[0], int(row[3]))),
                int(max(0, int(row[2]))):int(min(img.shape[1], int(row[4]))), :]
    # patch = img[int(row[2]):int(row[4]),
    #             int(row[1]):int(row[3]), :]
    return patch


def npy_from_textfile(text_file, patch_size, output,
                      nb_channels=3,
                      data_dir='/data/lisa/data/faces/GoogleEmotionDataset'):

    # get list of patches
    t_file = open(text_file, 'r')
    lines = t_file.read().splitlines()  # remove \n at EOL
    t_file.close()

    thres = 100000

    nb_patches = min(len(lines), thres)

    # Create hdf output
    out_shape = (nb_patches, patch_size * patch_size * nb_channels)
    out = np.zeros(out_shape)


    t0 = time()
    for cur, line in enumerate(lines):
        print cur, '/', nb_patches, line
        if (cur >= nb_patches):
            break

        patch = image_from_line(line, data_dir)
        #cv2.imshow('patch', patch)
        patch = cv2.resize(patch, (patch_size, patch_size),
                           interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('patchresize', patch)
        #cv2.waitKey(0)
        patch = patch.reshape(1, patch_size * patch_size * nb_channels)

        out[cur, :] = patch

    np.savetxt(output, out)


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: <text_file> <patch_size> <output_file>" % sys.argv[0])
        sys.exit(1)

    text_file = sys.argv[1]
    patch_size = int(sys.argv[2])
    output_file = sys.argv[3]
    print patch_size

    npy_from_textfile(text_file, patch_size, output_file)







