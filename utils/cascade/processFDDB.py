import sys
from os.path import join, isfile
from time import time
import cv2
# Custom imports
from facedet.utils.cascade.cascade import cascade


def process_fold(models, fprops, scales, sizes, strides,
                 probs, overlap_ratio,
                 nb_fold, out_dir,
                 min_pred_size=30,
                 piece_size=700,
                 max_img_size=1000,
                 fold_dir="/data/lisa/data/faces/FDDB/FDDB-folds/",
                 img_dir="/data/lisa/data/faces/FDDB/",
                 mode='rect',
                 max_det=60):
    """
    Apply the cascade of fprops to the folds
    Write the results at <out_dir>/fold-<nb_fold>-out.txt
    """

    # define file indicating list of files
    if nb_fold < 10:
        nb_s = "0" + str(nb_fold)
    else:
        nb_s = str(nb_fold)

    # Define fold
    fold = join(fold_dir, "FDDB-fold-"+nb_s+".txt")
    print "Working on", fold
    # define list of files as a pyList
    files = []
    with open(fold, "rb") as fold_list:
        for line in fold_list:
            files.append(join(img_dir, line[:-1]+".jpg"))  # Remove \n

    # Perform detection
    rois_tot = []
    scores_tot = []
    l_f = len(files)
    for i, f in enumerate(files[:l_f]):
        sys.stdout.write("\r" + str(nb_fold) + "th fold,"
                         + str(i) + "/" + str(l_f)
                         + " processed images | " + f)
        sys.stdout.flush()
        print ''

        # Perform cascade classification on image f
        if isfile(f):
            img_ = cv2.imread(f)
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
            cur_scales = []
            for i in range(len(scales)):
                cur_scales.append([e for e in scales[i]
                                   if sizes[i]/e >= min_pred_size])
            t0 = time()
            print img_.shape
            rois, scores = cascade(img_,
                                   models, fprops,
                                   cur_scales, sizes, strides,
                                   overlap_ratio, piece_size, probs)
            t = time()
            if need_rsz:
                # Rescale rois
                for i in range(len(rois)):
                    for j in range(len(rois[i])):
                        rois[i][j] = [er/rs for er in rois[i][j]]
            rois_tot.append(rois)
            scores_tot.append(scores)
        else:
            print f, 'does not exist'
            rois_tot.append([])
            scores_tot.append([])
    # Writing the results now

    output_fold = join(out_dir, "fold-"+nb_s+"-out.txt")
    # print output_fold
    with open(output_fold, 'wb') as output:
        for i, f in enumerate(files):
            # print "here:", output_fold
            # We need to format names to fit FDDB test
            # We remove /data/lisa/data/faces/FDDB and the extension
            n = f.split("/")[6:]
            n = "/".join(n)[:-4]
            m = min(max_det, len(rois_tot[i]))

            output.write(n+"\n")  # Filename for FDDB
            output.write(str(m)+'\n')  # len(rois_tot[i]))+"\n")  # Nb of faces
            # Faces for the image
            for roi, score in zip(rois_tot[i][:m], scores_tot[i][:m]):

                # <x0, y0, w, h, score>
                s = str(roi[0, 1]) + ' ' + str(roi[0, 0])+' '
                w = roi[1, 1] - roi[0, 1]
                h = roi[1, 0] - roi[0, 0]
                if mode == 'rect':
                    s += str(w) + ' ' + str(h) + ' '
                elif mode == 'ellipse':
                    s = str(w/2) + ' ' + str(h/2) + ' 0 ' + s
                output.write(s + str(score) + '\n')


