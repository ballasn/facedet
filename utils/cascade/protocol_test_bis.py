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
        new_scale = [0]*len(slices)
        crops = [0]*len(slices)
        # For each RoI of the past level
        for j, sl in enumerate(slices):
            # Compute the scale so the new region is equal to one
            crops[j] = img[sl]
            s = float(sizes[i]) / crops[j].shape[0]
            new_scale[j] = [s*e for e in scales[i] if
                    int(s*e*crops[j].shape[0]) >= sizes[i]]
            # Sort in ascending order
            new_scale[j].sort()
            # Sort in descending order
            new_scale[j] = new_scale[j][::-1]

        # Create and process the minibatches
        res = []
        for idx in range(len(slices)):
            res.append({})

        # Loop on the number of potential scales
        for i_s in range(len(scales[i])):
            valid_scale = False
            minibatch = None
            indices = []
            crop_size = 0
            for j, sc in enumerate(new_scale):
                # If the scale exists eg if the patch is big enough
                if i_s < len(sc):
                    if crop_size == 0:
                        crop_size = int(sc[i_s]*crops[j].shape[0])
                    valid_scale = True
                    indices.append(j)
                    new_crop = cv2.resize(crops[j],
                            (crop_size, crop_size),
                            interpolation=cv2.INTER_CUBIC)
                    # Transform to C01
                    new_crop = np.transpose(new_crop, (2, 0, 1))
                    # Transform to BC01
                    new_crop = np.reshape(new_crop, [1]+list(new_crop.shape))
                    # And add to the minibatch
                    if minibatch is None:
                        minibatch = np.copy(new_crop)
                    else:
                        minibatch = np.concatenate((minibatch,
                            np.copy(new_crop)))
            if not valid_scale:
                break
            size_example = minibatch.shape[1]*minibatch.shape[2]*minibatch.shape[3]
            max_size = 1*(10**7)
            # Size in examples of a chunk
            chunk_size = max_size/size_example
            # Number of chunks to bve produced
            chunk_nb = int(np.ceil(minibatch.shape[0]/float(chunk_size)))

            predictions = None
            for ch in range(chunk_nb):
                chunk = minibatch[ch*chunk_size:(ch+1)*chunk_size, :, :, :]
                if predictions is None:
                    predictions = fprops[i](chunk)
                else:
                    predictions = np.concatenate((predictions,
                                                  fprops[i](chunk)))
            for n, p_idx in enumerate(indices):
                try:
                    res[p_idx][new_scale[p_idx][i_s]] = \
                            np.copy(predictions[n, 0, :, :])
                except IndexError:
                    print len(res), p_idx
                    print len(new_scale), p_idx
                    print 'new_scale', new_scale
                    print 'indices', indices
                    print len(new_scale[p_idx]), i_s
                    exit(1)

        parent_idx = range(len(res))
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
        rois = correct_rois(rois, img.shape)
        scores = next_scores
        # Get the slices from the absolute values
        slices = rois_to_slices(rois)

    return rois, scores


