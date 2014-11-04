import cv2
import numpy as np
from theano import function
import theano.tensor as T
import cPickle as pkl
from utils.cascade.nms import fast_nms, nms_scale, dummy_nms
from utils.cascade.image_processing import process_image
from utils.cascade.rois import get_rois, correct_rois, rois_to_slices
from math import sqrt
import glob
def cascade(img, models, fprops, scales, sizes, strides, overlap_ratio, probs=None):
    """
    Perform the cascade classifier on image
    Returns the list of bounding boxes of the found items
    ---------------------------------------------
    image_file : path to an image
    fprops : list of fprop_funcs
                  will be used in the indexing order
    scales : list of lists of ints, define the scales used by each model
             will be used in the indexing order
    sizes : list of ints
            sizes of inputs for the corresponding classifiers
    strides : list of ints
              strides for the corresponding classifiers
    probs : list of floats
            if not None, define acceptance prob for each classifier
    """
    assert len(models) == len(fprops)
    assert len(fprops) == len(sizes)
    assert len(sizes) == len(strides)

    if probs is not None:
        assert len(strides) == len(probs)

    # Perform first level
    res = process_image(fprops[0], img, scales[0], sizes[0])
######################################################
    #res = fast_nms(res, sizes[0], strides[0], probs[0], overlap_ratio[0])
    res = dummy_nms(res, probs[0])
    #res = nms_scale(res, sizes[0], strides[0])
######################################################
    rois, scores = get_rois(res, models[0], enlarge_factor=0,
                            overlap_ratio=overlap_ratio[0],
                            remove_inclusion=(len(sizes) > 1))
    rois = correct_rois(rois, img.shape)
    slices = rois_to_slices(rois)

    for i in xrange(1, len(fprops)):

        next_rois = []
        next_scores = []

        # For each RoI of the past level
        for j, sl in enumerate(slices):
            crop_ = img[sl]
            # Change the actual scales used
            # if crop is smaller than predict size
            if crop_.shape[0] < sizes[i]:
                actual_scales = [float(crop_.shape[0])/float(sizes[i])]
            else:
                actual_scales = scales[i]
            res_ = process_image(fprops[i], crop_, actual_scales, sizes[i])
######################################################
            # Threshlod on the cumylated score like the Soft Cascade
            #res_ = fast_nms(res_, sizes[i], strides[i], probs[i]-scores[j)
            res_ = dummy_nms(res_, probs[i]-scores[j])
######################################################

            local_rois, local_scores = get_rois(res_, models[i],
                                                enlarge_factor=0.3,
                                                overlap_ratio=overlap_ratio[i],
                                                remove_inclusion=(len(sizes) > i+1))
            local_rois = correct_rois(local_rois, crop_.shape)

            # Get the absolute coords of the new RoIs
            # The score is now the sum of
            # the local score and the score of the RoIs
            for r, s in zip(local_rois, local_scores):
                next_rois.append(r + rois[j][0, :])
                next_scores.append(s + scores[j])
        rois = next_rois
        scores = next_scores
        # Get the slices from the absolute values
        slices = rois_to_slices(rois)
    return rois, scores


if __name__ == "__main__":
    # Define image and model
    img_file =\
    '/data/lisa/data/faces/FDDB/2002/07/24/big/img_402.jpg'
    #l = glob.glob('/u/chassang/Pictures/face_det/*.bmp')
    l = [img_file]
    model_file1 =\
    '/data/lisatmp3/chassang/facedet/models/16/dsn16_700k_best.pkl'
    # model_file2 = '../../exp/convtest/models/conv48_best.pkl'
    # model_file3 = '../../exp/convtest/convTest96_best.pkl'

    imgs = [cv2.imread(img_file) for img_file in l]

    # Define predictor
    with open(model_file1, 'r') as m_f:
        model1 = pkl.load(m_f)
    #model1.layers.pop()
    # Compile network prediction function
    x = T.tensor4("x")
    predict1 = function([x], model1.fprop(x))

    # Define parameters
    models = [model1]
    fprops = [predict1]
    sizes = [16]
    strides = [2]
    scales = [0.05 * sqrt(2)**e for e in range(5)  ]
    local_scales = [scales]
    probs = [0.7]
    overlap_ratio = [0.3]
    print 'local_scales', local_scales

    for j,img in enumerate(imgs):
        rois, scores = cascade(img, models, fprops, local_scales, sizes,
                strides,overlap_ratio, probs)
        print 'ok with image', j
        for i in xrange(len(rois)):
            cv2.rectangle(img, (int(rois[i][0, 1]), int(rois[i][0, 0])),
                          (int(rois[i][1, 1]), int(rois[i][1, 0])),
                            (0, 255, 0), 2)
            #cv2.imwrite('./images/processed_'+str(j)+'.bmp',img)
        cv2.imshow('r3', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #print rois
    #print scores
    '''
    # Flip the image vertically
    img2 = np.copy(img[::-1, :])
    rois2, scores2 = cascade(img2, models, fprops, local_scales, sizes,
                             strides, overlap_ratio, probs)

    # Flip the image horizontally
    img3 = np.copy(img[:, ::-1, :])
    rois3, scores3 = cascade(img3, models, fprops, local_scales, sizes,
                             strides, overlap_ratio, probs)
    # Display results as squares on the image

                      (0, 255, 0), 2)
    for i in xrange(len(rois)):
        cv2.putText(img, str(scores[i]),
                    (int(rois[i][0, 1]) + 10, int(rois[i][0, 0]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0))
    cv2.imshow('r', img)
    for i in xrange(len(rois2)):
        cv2.rectangle(img2, (int(rois2[i][0, 1]), int(rois2[i][0, 0])),
                      (int(rois2[i][1, 1]), int(rois2[i][1, 0])),
                      (0, 255, 0), 2)
    for i in xrange(len(rois2)):
        cv2.putText(img2, str(scores2[i]),
                    (int(rois2[i][0, 1]) + 10, int(rois2[i][0, 0]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0))

    cv2.imshow('r2', img2)
    for i in xrange(len(rois3)):
        cv2.rectangle(img3, (int(rois3[i][0, 1]), int(rois3[i][0, 0])),
                      (int(rois3[i][1, 1]), int(rois3[i][1, 0])),
                      (0, 255, 0), 2)
    for i in xrange(len(rois3)):
        cv2.putText(img3, str(scores3[i]),
                    (int(rois3[i][0, 1]) + 10, int(rois3[i][0, 0]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0))
    '''
