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


def cascade(img, models, fprops,
            scales, sizes, strides,
            overlap_ratio, probs=None):
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
    res = dummy_nms([res], probs[0], [0], [0])
    rois, scores = get_rois(res, models[0],
                            enlarge_factor=0,
                            overlap_ratio=overlap_ratio[0],
                            remove_inclusion=False)
    #                            remove_inclusion=(len(sizes) > 1))
    rois = correct_rois(rois, img.shape)
    slices = rois_to_slices(rois)


    for i in xrange(1, len(fprops)):
        next_rois = []
        next_scores = []
        res = []
        parent_idx = []
        # For each RoI of the past level
        for j, sl in enumerate(slices):
            ## Compute the scale so the new region is equal to one
            crop_ = img[sl]
            s = float(sizes[i]) / crop_.shape[0]
            new_scale = [s*e for e in scales[i]]
            res.append(process_image(fprops[i], crop_,
                                     new_scale, sizes[i]))
            parent_idx.append(j)

        # Filter ouput maps < p
        res = dummy_nms(res, probs[i], parent_idx, scores)
        # Get Rois from output maps
        next_rois, next_scores = get_rois(res, models[i],
                                          prev_rois=rois,
                                          prev_score=scores,
                                          enlarge_factor=0.1,
                                          overlap_ratio=overlap_ratio[i],
                                          remove_inclusion=False)
                                            #(len(sizes) > i+1))


        rois = next_rois
        scores = next_scores
        # Get the slices from the absolute values
        slices = rois_to_slices(rois)

    return rois, scores


