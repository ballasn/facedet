from time import time
import cv2
from facedet.utils.cascade.optim import process_in_chunks, thresholding
from facedet.utils.cascade.nms import dummy_nms
from facedet.utils.cascade.image_processing import process_image, cut_in_dict
from facedet.utils.cascade.rois import get_rois, correct_rois, rois_to_slices, get_input_coords


def cascade(img, models, fprops,
            scales, sizes, strides,
            overlap_ratio, piece_size,
            probs=None):
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

    t_fprop = 0
    t_NMS = 0
    nb_slices = 0

    # Perform first level
    res = []
    for s in scales[0]:
        # Define the input : a resized image
        img_test = cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s)),
                              interpolation=cv2.INTER_CUBIC)

        if max(img_test.shape[0:2]) > piece_size:
            # Define the pieces
            coords = []
            examples = []
            pieces = cut_in_dict(img_test, strides[0], sizes[0], piece_size)

            # pieces[top, left] = array[ x, y, c]
            for (ix, iy) in pieces:
                coords.append([ix, iy])
                examples.append(pieces[(ix, iy)])

            # Process in batch mode
            t_1 = time()
            res_ = process_in_chunks(fprops[0], examples, max_size=10**6)
            t_2 = time()
            t_fprop += t_2-t_1

            for i in range(len(res_)):
                res_[i] = thresholding(res_[i], probs[0], s)

            st_ = float(strides[0])
            for i in range(len(res_)):
                [ix, iy] = coords[i]
                for j in range(len(res_[i])):
                    # Coords in local feature maps -> coords in general feature
                    # map
                    res_[i][j][1] = res_[i][j][1] + ix / st_
                    res_[i][j][2] = res_[i][j][2] + iy / st_
                    # Test for coords
                    cs = get_input_coords(res_[i][j][1], res_[i][j][2], models[0])
                    if max(cs[0][0], cs[1][0]) > img_test.shape[0] or\
                       max(cs[0][1], cs[1][1]) > img_test.shape[1]:
                        print ''
                        print res_[i][j]
                        print cs
                        print 'problem line 80'
                        exit(34)

            # Gather the results in one list
            for j in range(1, len(res_)):
                res_[0].extend(res_[j])
            res_ = res_[0]

        else:
            t_1 = time()
            res_ = process_image(fprops[0], img_test, [1.0],
                                 strides[0], sizes[0], piece_size)
            t_2 = time()
            t_fprop += t_2-t_1

            res_ = dummy_nms([res_], probs[0], [0], [0])
            for i in range(len(res_)):
                res_[i][0] = s
                if res_[i][1] > img.shape[0] or\
                   res_[i][2] > img.shape[1]:
                        print 'Out of bounds !'
                        print ix, iy
                        print s, st_
                        print res[i][j]
                        print img.shape
                        exit(2)
        res.extend(res_)

    res.sort(key=lambda x: x[3], reverse=True)
    t_1 = time()
    rois, scores = get_rois(res, models[0], enlarge_factor=0,
                            overlap_ratio=overlap_ratio[0],
                            remove_inclusion=False)
    t_2 = time()
    t_NMS += t_2-t_1
    nb_slices += len(rois)

    # Needed in case we enlarge detection areas
    '''
    for e in rois:
        print img.shape
        print e
        cv2.rectangle(img,
                      (int(e[0, 1]), int(e[0, 0])),
                      (int(e[1, 1]), int(e[1, 0])),
                      (0, 255, 0), 2)
        disp = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
        cv2.imshow('a', disp)
        cv2.moveWindow('a', 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
    rois = correct_rois(rois, img.shape)
    slices = rois_to_slices(rois)
############################################################

    # Now loop on other models
    for i in xrange(1, len(fprops)):
        next_rois = []
        next_scores = []
        res = []
        parent_idx = []
        new_scale = [0]*len(slices)
        crops = [0]*len(slices)

        # For each RoI of the past level
        for j, sl in enumerate(slices):
            crops[j] = img[sl]

            if crops[j].shape[0] == 0:
                print 'Problem with the size of the crop'
                print crops[j].shape
                exit(1)

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
            minibatch = []
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
                    # And add to the minibatch
                    minibatch.append(new_crop)
            if not valid_scale:
                break

            # Process the minibatch
            t_1 = time()
            res_ = process_in_chunks(fprops[i], minibatch, max_size=10**6)
            t_2 = time()
            t_fprop += t_2-t_1

            for n, p_idx in enumerate(indices):
                try:
                    res[p_idx][new_scale[p_idx][i_s]] = res_[n]
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
        t_1 = time()
        next_rois, next_scores = get_rois(res, models[i],
                                          prev_rois=rois,
                                          prev_score=scores,
                                          enlarge_factor=0.1,
                                          overlap_ratio=overlap_ratio[i],
                                          remove_inclusion=False)
        t_2 = time()
        t_NMS += t_2-t_1
        rois = next_rois
        rois = correct_rois(rois, img.shape)
        scores = next_scores
        # Get the slices from the absolute values
        slices = rois_to_slices(rois)
    print 'Writing results'
    s = str(t_NMS)+' '+str(t_fprop)+' '+str(img.shape[0]*img.shape[1])
    s += ' '+str(nb_slices)+'\n'
    with open('NMS_pixels.txt', 'a') as curve:
        curve.write(s)
        '''
    for e in rois:
        cv2.rectangle(img,
                      (int(e[0, 1]), int(e[0, 0])),
                      (int(e[1, 1]), int(e[1, 0])),
                      (0, 0, 255))
        cv2.imshow('a', img)
        cv2.moveWindow('a', 200, 200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    return rois, scores
