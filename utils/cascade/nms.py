import copy
import sys
import numpy as np
from time import time
from scipy import ndimage
import matplotlib.pyplot as plt


def overlap(a, b):
    """
    For rectangles defined by 2 points
    """
    return not(a[2]<=b[0] or a[3]<=b[1] or a[0]>=b[2] or a[1]>=b[3])


def overlap_scale(a, b, size, stride):
    x0 = a[1]*stride/a[0]
    y0 = a[2]*stride/a[0]
    x1 = b[1]*stride/b[0]
    y1 = b[2]*stride/b[0]
    pa = [x0, y0, x0 + size/a[0], y0 + size/a[0]]
    pb = [x1, y1, x1 + size/b[0], y1 + size/b[0]]
    return overlap(pa, pb)


def fast_nms(maps, size, stride, prob, overlap_ratio):
    """
    Perform NMS scale by scale first using filters from scipy
    Then perform spatial pooling
    Patch are supposed to be square
    ---------------------------------------------------------
    maps : dict of the features maps indexed by scales
    size : size of a patch
    stride : stride between patches
    """
    rval = []
    # Define overlapping region
    # The filter has size over,over and is centered on a point
    over = 2 * ((size - 1) / stride) + 1
    over = int(over * (1.0 - overlap_ratio))
    for s in maps:
        # Apply the filter to the overlapping region
        maps[s] = maps[s] * (maps[s] > prob)
        maxF = ndimage.filters.maximum_filter(maps[s], (over, over))
        maps[s] = maps[s] * (maps[s] == maxF)
        n_z = np.transpose(np.nonzero(maps[s]))
        rval.extend([[s, n_z[e, 0], n_z[e, 1], maps[s][n_z[e, 0], n_z[e, 1]]]
                     for e in range(len(n_z))])
    return rval


def dummy_nms(list_maps, prob, parent_idx, scores):# slice_idx=0):
    """
    Keep all a elements > probs
    ---------------------------------------------------------
    maps : dict of the features maps indexed by scales
    size : size of a patch
    stride : stride between patches
    """
    rval = []
    for slice_idx, maps in zip(parent_idx, list_maps):
        for s in maps:
            # Apply the filter based on proba
            maps[s] = maps[s] * (maps[s] > prob - scores[slice_idx])
            n_z = np.transpose(np.nonzero(maps[s]))
            rval.extend([[s,
                          n_z[e, 0], n_z[e, 1],
                          maps[s][n_z[e, 0], n_z[e, 1]],
                          slice_idx]
                         for e in range(len(n_z))])
            #print 'nb of nonzero patches :', len(rval)
    if rval != []:
        rval.sort(key=lambda x: x[3], reverse=True)
        #print 'min :', min(rval, key=lambda x: x[3])
        #print'max :', max(rval, key=lambda x: x[3])
    return rval


def nms_scale(nz, size, stride):
    """
    Perform the NMS between scales
    """
    # Perform NMS between scales
    for i, e in enumerate(nz):
        if e is None:
            continue
        [s, x, y, sc] = e
        for j, f in enumerate(nz):
            if f is None or j == i:
                continue
            [s_t, x_t, y_t, sc_t] = f
            if overlap_scale([s, x, y], [s_t, x_t, y_t], size, stride):
                if sc >= sc_t:
                    nz[j] = None
                else:
                    nz[i] = None
                    break  # Go to i+1
    res = []
    for e in nz:
        if e is None:
            continue
        res.append(e)
    return res

