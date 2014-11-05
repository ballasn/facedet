from utils.cascade.AFW.nmsAFW import dummy_nms
from utils.cascade.AFW.image_processingAFW import process_image, cut_in_dict
from utils.cascade.AFW.roisAFW import get_rois, correct_rois, rois_to_slices
import numpy as np

def cascade(img, models, fprops, scales, sizes, strides, overlap_ratio,
            piece_size, probs=None):
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
    # Cut the image in pieces if needed
    if max(img.shape) > piece_size:
        pieces = cut_in_dict(img, strides[0], sizes[0], piece_size)
        # pieces[top, left] = array[ x, y, c]
        print 'using scales from', min(scales[0]), 'to', max(scales[0]),
        print 'on', len(pieces.keys()), 'pieces'
        res = []
        for (ix, iy) in pieces:
            print (ix, iy),
            sh = pieces[(ix, iy)].shape
            print sh
            example = img[ix:ix+sh[0], iy:iy+sh[1], :]
            res_ = process_image(fprops[0], example, scales[0],
                                 strides[0], sizes[0], piece_size)
            res_ = dummy_nms(res_, probs[0])
            # Move the rois to their absolute position
            for i in range(len(res_)):
                res_[i][1] = res_[i][1] + ix*res_[i][0]/float(strides[0])
                res_[i][2] = res_[i][2] + iy*res_[i][0]/float(strides[0])
            res.extend(res_)
        res.sort(key=lambda x: x[3], reverse=True)
        rois, scores = get_rois(res, models[0], enlarge_factor=0,
                                overlap_ratio=overlap_ratio[0],
                                remove_inclusion=(len(sizes) > 1))

        rois = correct_rois(rois, img.shape)
        if len(rois)==0:
            print scores
            print rois
            print res
            exit(1)

    else:
        res_ = process_image(fprops[0], img, scales[0],
                            strides[0], sizes[0], piece_size)
        # res = fast_nms(res, sizes[0], strides[0], probs[0], overlap_ratio[0])
        res = dummy_nms(res_, probs[0])
        if len(res)==0:
            print 'No res after NMS !'
            print res
            print 'Maps before NMS :'
            for s in res_:
                print 'scale :', s, 'min value', np.amin(res_[s]),
                print 'max value', np.amax(res_[s])
            exit(1)
        # res = nms_scale(res, sizes[0], strides[0])
        rois, scores = get_rois(res, models[0], enlarge_factor=0,
                                overlap_ratio=overlap_ratio[0])
        if len(rois)==0:
            print scores
            print rois
            print res
            exit(1)
        rois = correct_rois(rois, img.shape)
        if len(rois)==0:
            print scores
            print rois
            print res
            exit(1)
        slices = rois_to_slices(rois)
    '''
    if len(rois) == 0:
        print rois
        print scores
        print 'No detection on this image, this should not happen'
        exit(3, 'len(rois) ==0')
    '''
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
            res_ = process_image(fprops[i], crop_, actual_scales,
                                 strides[i], sizes[i], piece_size)
            # Threshlod on the cumulated score like the Soft Cascade
            res_ = dummy_nms(res_, probs[i]-scores[j])

            local_rois, local_scores = get_rois(res_, models[i],
                                                enlarge_factor=0.3,
                                                overlap_ratio=overlap_ratio[i])
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

    assert len(rois) == len(scores)
    assert len(rois) > 0
    for e in rois:
        if e[1, 0] > img.shape[0] or e[1, 1] > img.shape[1]:
            print e
            print img.shape
            exit(1, 'Out of bounds')
    return rois, scores
