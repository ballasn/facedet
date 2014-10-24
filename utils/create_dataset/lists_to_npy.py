import sys
import os
import cv2
import numpy as np
from time import time
import numpy as np
import random
import math
from random import shuffle



def image_from_line(line,
                    data_dir='/data/lisa/data/faces/GoogleEmotionDataset'):
    """
    Return the corresponding patch from a line of text
    --------------------------------------------------
    line : string defining a patch as
           file x0 y0 x1 y1
    """
    #print line
    row = line.strip().split(' ')
    filepath = os.path.join(data_dir, row[0])
    img = cv2.imread(filepath)
    for i in xrange(1, 5):
        row[i] = int(np.floor(float(row[i])))
    #print row, img.shape
    #cv2.imshow('Img', img)
    #print max(0, row[2]), min(img.shape[0], row[4])
    #print max(0, row[1]), min(img.shape[1], row[3])
    lx = row[3] - row[1]
    ly = row[4] - row[2]
    #print lx, ly, lx /2, ly / 2
    if ly > lx :
        row[3] = row[3] + ly /2
        row[1] = row[1] - ly /2
    else:
        row[4] = row[4] + ly /2
        row[2] = row[2] - ly /2

    patch = img[int(max(0, int(row[2]))):int(min(img.shape[0], int(row[4]))),
                int(max(0, int(row[1]))):int(min(img.shape[1], int(row[3]))), :]

    ### Warning need to be inverted for FaceDataset
    patch = img[int(max(0, int(row[1]))):int(min(img.shape[0], int(row[3]))),
                int(max(0, int(row[2]))):int(min(img.shape[1], int(row[4]))), :]
    # patch = img[int(row[2]):int(row[4]),
    #             int(row[1]):int(row[3]), :]
    return patch

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    print ""
    print w, h
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat,
                          (int(math.ceil(nw)), int(math.ceil(nh))),
                          flags=cv2.INTER_LANCZOS4)

def image_from_line_augmented(line,
                              data_dir='/data/lisa/data/faces/GoogleEmotionDataset'):
    """
    Return the corresponding patch from a line of text
    --------------------------------------------------
    line : string defining a patch as
           file x0 y0 x1 y1
    """
    #print line
    row = line.strip().split(' ')
    filepath = os.path.join(data_dir, row[0])
    img = cv2.imread(filepath)
    for i in xrange(1, 5):
        row[i] = int(np.floor(float(row[i])))
    #print row, img.shape
    #cv2.imshow('Img', img)
    #print max(0, row[2]), min(img.shape[0], row[4])
    #print max(0, row[1]), min(img.shape[1], row[3])




    #patch = img[int(max(0, int(row[2]))):int(min(img.shape[0], int(row[4]))),
    #            int(max(0, int(row[1]))):int(min(img.shape[1], int(row[3]))), :]
    patch = img[int(max(0, int(row[1]))):int(min(img.shape[0], int(row[3]))),
                int(max(0, int(row[2]))):int(min(img.shape[1], int(row[4]))), :]
    #cv2.imshow('patch_orig', patch)

    lengthx = row[3] - row[1]
    lengthy = row[4] - row[2]
    length = max(row[3] - row[1], row[4] - row[2])

    ### First we augment the size of the bounding box
    row[1] -= (row[3] - row[1]) / 2
    row[3] += (row[3] - row[1]) / 2
    row[2] -= (row[4] - row[2]) / 2
    row[4] += (row[4] - row[2]) / 2

    lx = row[3] - row[1]
    ly = row[4] - row[2]

    ### Get a square box
    if ly > lx :
        row[3] = row[3] + ly / 9
        row[1] = row[1] - ly / 9
    else:
        row[4] = row[4] + lx / 9
        row[2] = row[2] - lx / 9

    lx = row[3] - row[1]
    ly = row[4] - row[2]




    ### Extract patches
    ### Warning need to be inverted for FaceDataset
    #patch = img[int(max(0, int(row[1]))):int(min(img.shape[0], int(row[3]))),
    #            int(max(0, int(row[2]))):int(min(img.shape[1], int(row[4]))), :]
    patch = img[int(max(0, int(row[1]))):int(min(img.shape[0], int(row[3]))),
                int(max(0, int(row[2]))):int(min(img.shape[1], int(row[4]))), :]

    patch = np.copy(patch)
    ### Flip at random
    # if random.choice([True, False]):
    #     patch =  patch[:, ::-1, :]


    #cv2.imshow('patch', patch)

    torot = patch
    # ### Apply a random rotation
    if random.choice([True, False]):
        patch = rotate_about_center(torot, np.random.sample() * 40)
    else:
        patch = rotate_about_center(torot, np.random.sample() * -40)
    #cv2.imshow('patch_rot', patch)



    ### Do a random (small) translation
    if random.choice([True, False]):
        tx = np.random.sample() * 0.1 * lengthx
    else:
        tx = np.random.sample() * 0.1 * -lengthx
    if random.choice([True, False]):
        ty = np.random.sample() * 0.1 * lengthy
    else:
        ty = np.random.sample() * 0.1 * -lengthy
    #tx = 0
    #ty = 0


    ### Return a patch around the center
    cx = patch.shape[0] / 2
    cy = patch.shape[1] / 2
    patch = patch[int(max(0, cx-length/2+tx)):int(min(img.shape[0], cx+length/2+tx)),
                  int(max(0, cy-length/2+ty)):int(min(img.shape[1], cy+length/2+ty)), :]

    #cv2.imshow('patch_aug', patch)
    #cv2.waitKey(0)
    return patch


def npy_from_textfile(text_file, patch_size, output, dup=1,
                      nb_channels=3,
                      data_dir='/data/lisa/data/faces/GoogleEmotionDataset'):

    # get list of patches
    t_file = open(text_file, 'r')
    lines = t_file.read().splitlines()  # remove \n at EOL
    t_file.close()

    #thres = 700000

    nb_patches = dup * len(lines)

    # Create output
    out_shape = (nb_patches, patch_size * patch_size * nb_channels)
    out = np.zeros(out_shape, dtype='float32')


    for d in xrange(0, dup):
        t0 = time()
        for cur, line in enumerate(lines):

            sys.stdout.write('\r'+str(cur)+' / '+str(nb_patches)+' '+line)
            sys.stdout.flush()
            if (cur >= nb_patches):
                break

            if dup == 0:
                patch = image_from_line(line, data_dir)
            else:
                patch = image_from_line_augmented(line, data_dir)
            #cv2.imshow('patch', patch)
            patch = cv2.resize(patch, (patch_size, patch_size),
                               interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('patchresize', patch)
            #cv2.waitKey(0)
            patch = np.asarray(patch, dtype='float32')
            patch = patch.reshape(1, patch_size * patch_size * nb_channels)
            out[cur, :] = patch

    np.save(output, out)


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage %s: <text_file> <patch_size> <output_file> [dup]" % sys.argv[0])
        sys.exit(1)

    text_file = sys.argv[1]
    patch_size = int(sys.argv[2])
    output_file = sys.argv[3]


    if sys.argv == 5:
        dup = int(sys.arg[4])
    else:
        dup = 1


    npy_from_textfile(text_file, patch_size, output_file, dup,
                      data_dir='')







