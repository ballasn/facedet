import sys
import os
import cv2
import numpy as np
from time import time
import tables
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
    patch = img[int(row[1]):int(row[3]),
                int(row[2]):int(row[4]), :]
    return patch


def hdf_from_textfile(text_file, patch_size, output,
                      nb_channels=3):

    # get list of patches
    t_file = open(text_file, 'r')
    lines = t_file.read().splitlines()  # remove \n at EOL
    t_file.close()
    tot = len(lines)

    # Create hdf output
    f = tables.openFile(output, 'w')
    atom = tables.Float32Atom()
    # No compression for now
    filters = tables.Filters(complib='blosc', complevel=0)
    hdf_shape = (tot, patch_size * patch_size * nb_channels)
    data = f.createCArray(f.root, 'denseFeat', atom,
                          hdf_shape, filters=filters)
    t0 = time()
    for cur, line in enumerate(lines):
        if cur % 100 == 0:
            t = time()
            sys.stdout.write("\r"+str(cur)+"/"+str(tot)+' ' +
                             str(t-t0)+' s for the last 100 patches')
            sys.stdout.flush()
            t0 = time()

        patch = image_from_line(line)
        patch = cv2.resize(patch, (patch_size, patch_size),
                           interpolation=cv2.INTER_CUBIC)
        patch = patch.reshape(1, patch_size * patch_size * nb_channels)
        data[cur, :] = patch

    f.flush()
    f.close()


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: <text_file> <patch_size> <output_file>" % sys.argv[0])
        sys.exit(1)

    text_file = sys.argv[1]
    patch_size = int(sys.argv[2])
    output_file = sys.argv[3]
    print patch_size

    hdf_from_textfile(text_file, patch_size, output_file)







