'''
Functions to process FDDB and Pascal/AFLW datasets
These will write the results in the correct format to get scores with the
execs.
'''
import sys
from os.path import join, isfile
import cv2
# Custom imports
from facedet.utils.cascade.cascade import cascade
from time import time


def process_pascal(models, fprops, scales, sizes, strides, probs,
                   overlap_ratio,
                   out_dir,
                   min_pred_size=30,
                   piece_size=700.0,
                   max_img_size=1000.0,
                   id_list="/data/lisa/data/faces/AFW/annot/list",
                   img_dir="/data/lisa/data/faces/AFW/testimages/",
                   max_det=30,
                   name=''):
    """
    Apply the cascade of fprops to the folds
    Write the results at <out_dir>/res-out.txt in pascal VOC format
    ---------------------------------------------------------------
    min_pred_size : min size of a detection
    piece_size : size of pieces when we cut the image
    max_img_size : maximum size allowed as input,
                   if larger, the input is resized
    """

    # Read imgs list
    files = []
    ids = []
    with open(id_list) as _f:
        for line in _f:
            ids.append(line[:-1])  # Remove \n
            files.append(join(img_dir, line[:-1]+".jpg"))  # Remove \n

    # Perform detection
    rois_tot = []
    scores_tot = []
    l_f = len(files)
    for i, f in enumerate(files):
        sys.stdout.write(str(i) + "/" + str(l_f) + " processed images | " + f)
        sys.stdout.flush()
        # Perform cascade classification on image f
        if isfile(f):
            img_ = cv2.imread(f)
            print ''
            print 'Image shape :', img_.shape,

            # If max dim > max_size, resize
            need_rsz = max(img_.shape[0], img_.shape[1]) > max_img_size
            if need_rsz:
                print 'resized to',
                if img_.shape[0] > img_.shape[1]:
                    rs = float(max_img_size) / img_.shape[0]
                else:
                    rs = float(max_img_size) / img_.shape[1]
            # Warning : we need to do indicate reversed shapes with cv2 !!!
                img_ = cv2.resize(img_,
                                  (int(rs * img_.shape[1]),
                                   int(rs * img_.shape[0])),
                                  interpolation=cv2.INTER_CUBIC)
                print img_.shape,
            cur_scales = []
            for i in range(len(scales)):
                cur_scales.append([e for e in scales[i]
                                   if sizes[i]/e >= min_pred_size])

            # Perform cascade
            t0 = time()
            e = cascade(img_, models, fprops, cur_scales, sizes,
                        strides, overlap_ratio, piece_size, probs)
            if e != 2:
                rois, scores = e
            else:
                print f
                exit(2)

            t = time()
            print t-t0, 's on the image or',
            print img_.shape[0]*img_.shape[1]/(t-t0), 'pixels per s'
            if need_rsz:
                # Rescale rois
                for i in range(len(rois)):
                    for j in range(len(rois[i])):
                        rois[i][j] = [er/rs for er in rois[i][j]]
            rois_tot.append(rois)
            scores_tot.append(scores)
        else:
            # File does not exist
            print 'Warning : File not found'
            print f
            rois_tot.append([])
            scores_tot.append([])

    # Writing detections in PascalVOC format
    if name == '':
        output_filename = 'Min'+str(min_pred_size)+'Img'
        output_filename += str(int(max_img_size))
        output_filename += 'Sc'+str(round(min(scales[0]), 2))
        output_filename += '-'+str(round(max(scales[0]), 2))+'.txt'
    else:
        output_filename = name+'.txt'
    outputf = join(out_dir, output_filename)
    i = 1
    while isfile(outputf):
        outputf = outputf[:-4]+str(i)+'.txt'
        i += 1

    print 'Writing at', outputf
    with open(outputf, 'w') as output:
        for i, f in enumerate(files):
            if max_det != -1:
                m = min(max_det, len(rois_tot[i]))
            else:
                m = len(rois_tot[i])

            for roi, score in zip(rois_tot[i][:m], scores_tot[i][:m]):

                # <img_id score left top right bottom>
                x0 = int(roi[0, 0])
                y0 = int(roi[0, 1])
                x1 = int(roi[1, 0])
                y1 = int(roi[1, 1])

                output.write(ids[i] + ' ' + str(score) + ' ')
                # For AFW we have to flip the coordinates
                output.write(str(y0) + ' ' + str(x0) + ' ' + str(y1) + ' ' +
                             str(x1) + '\n')
