import sys
import os
import cPickle
import gzip
import cv2
from itertools import islice

from theano import tensor as T
from theano import config
from theano import function
import numpy as np


from models.layer.convVariable import ConvElemwise as Cv_new_path
from utils.layer.convVariable import ConvElemwise as Cv_old_path
from models.layer.convVariable import ConvElemwise as Cv_new_path
from models.layer.corrVariable import CorrMMElemwise as Corr_new_path
from models.layer.cudnnVariable import CudNNElemwise as Cv_cudnn

### FIXME import dataset

from numpy import unravel_index


def oneofc(pred, nb_classes):
    opred = zeros((pred.shape[0], nb_classes))
    for i in  pred.shape[0]:
        opred[i, pred[i]] =1
    return opred

def oneofc_inv(pred):
    out = np.argmax(pred, axis=1)
    return out


def get_input_coords(i, j, model):
    """
    Returns the coords in the original input given the ones
    at the output
    -----------------------------
    model : model including conv layers
    """
    x0 = i
    y0 = j
    x1 = 1
    y1 = 1
    for l in model.layers[::-1]:
        if isinstance(l, Cv_new_path) or isinstance(l, Cv_old_path) or isinstance(l, Corr_new_path) or isinstance(l, Cv_cudnn):
            if l.pool_type is not None:
                x0 *= l.pool_stride[0]
                y0 *= l.pool_stride[0]
                x1 = (x1-1) * l.pool_stride[0] + l.pool_shape[0]
                y1 = (y1-1) * l.pool_stride[0] + l.pool_shape[0]

            x0 *= l.kernel_stride[0]
            y0 *= l.kernel_stride[0]
            x1 = (x1-1) * l.kernel_stride[0] + l.kernel_shape[0]
            y1 = (y1-1) * l.kernel_stride[0] + l.kernel_shape[0]
    return [x0, y0, x0+x1, y0+y1]

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
    #patch = img[int(max(0, int(row[1]))):int(min(img.shape[0], int(row[3]))),
    #            int(max(0, int(row[2]))):int(min(img.shape[1], int(row[4]))), :]
    # patch = img[int(row[2]):int(row[4]),
    #             int(row[1]):int(row[3]), :]
    return filepath, patch


if __name__ == '__main__':

    if len(sys.argv) < 6:
        print("Usage %s: list model layer feat out" % sys.argv[0])
        exit(1)


    layer = int(sys.argv[3])
    feat = int(sys.argv[4])
    patch_size = 96

    nb_run_model = 1
    ### Load model
    model = None
    with open(sys.argv[2], "rb") as file:
        model = cPickle.load(file)
    print(model)
    ### Pop the layers that we do not need
    for i in xrange(len(model.layers), layer, -1):
        print i
        model.layers.pop()

    print model

    ### Compile network prediction function
    x = T.tensor4("x")
    ### Model pop
    predict = function([x], model.fprop(x))


    ### Read input list
    patches = np.zeros((10, 5))
    filenames = ["tmp", "tmp", "tmp", "tmp", "tmp",
                 "tmp", "tmp", "tmp", "tmp", "tmp"]# ,
                 # "tmp", "tmp", "tmp", "tmp", "tmp",
                 # "tmp", "tmp", "tmp", "tmp", "tmp",
                 # "tmp", "tmp", "tmp", "tmp", "tmp",
                 # "tmp", "tmp", "tmp", "tmp", "tmp"]


    # get list of patches
    t_file = open(sys.argv[1], 'r')
    lines = t_file.read().splitlines()  # remove \n at EOL
    t_file.close()

    for cur, line in enumerate(lines):
        if cur > 10000:
            break
        filename, patch = image_from_line(line, "")
        #cv2.imshow("patch", patch)
        #cv2.waitKey(0)
        patch = cv2.resize(patch, (patch_size, patch_size),
                           interpolation=cv2.INTER_CUBIC)
        patch = np.reshape(patch, (1,) + patch.shape)
        patch = np.transpose(patch, (0, 3, 1, 2))
        score_map = predict(patch)
        val = np.asarray(score_map[0, feat - 1, :, :],
                         dtype=config.floatX)
        i, j = unravel_index(val.argmax(), val.shape)
        print cur, "/", len(lines), ":", i, j, val[i, j]
        val = val[i, j]
        find = False
        k = 0
        while not find and k < patches.shape[0]:
            if val > patches[k, 4]:
                coords = get_input_coords(i, j, model)
                patches[k, 0:4] = coords[:]
                filenames[k] = line
                #print k, b, feat - 1, i, j, val
                patches[k, 4] = val
                find = True
                #print "Here"
            k +=1


    ### Write output infos
    np.save("patch.npy", patches)
    with open(os.path.join("list.txt"), 'w') as fd:
        for item in filenames:
            fd.write("%s\n" % item.strip())


    for i in xrange(len(patches)):
        #frame = cv2.imread(filenames[i])
        filename, frame = image_from_line(filenames[i], "")
        #cv2.imshow("patch", patch)
        #cv2.waitKey(0)
        frame = cv2.resize(frame, (patch_size, patch_size),
                           interpolation=cv2.INTER_CUBIC)
        x0, y0, x1, y1, score = patches[i, :]

        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 255, 0), 2)
        cv2.putText(frame, "%.2f" % score,
                    (int(x0 + 5),  int(y0 + 10)),
                    cv2.FONT_HERSHEY_PLAIN, 0.5,
                    (0, 255, 0))
        cv2.imwrite(sys.argv[5] + "_" + str(i) + ".jpg", frame)
        #cv2.imshow('frame' + "_" + str(i) + ".jpg" ,frame)

    #cv2.waitKey(0)

