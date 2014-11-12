import sys
import cv2
from os.path import join
import cPickle as pkl
import theano.tensor as T
from theano import function
import importlib

# Custom imports
from facedet.utils.cascade.AFW.cascadeAFW import cascade

if __name__ == '__main__':

    if len(sys.argv) < 6:
        print "usage %s : <model> <img_list> <base_dir> <out_file> <params.py>"\
              % sys.argv[0]
        sys.exit(2)

    model_file = sys.argv[1]
    list_file = sys.argv[2]
    base_dir = sys.argv[3]
    out_file = sys.argv[4]

    # Define detector
    with open(model_file, 'r') as m_f:
        model = pkl.load(m_f)
    model.layers.pop()
    x = T.tensor4('x')
    detector = function([x], model.fprop(x))
    models = [model]
    fprops = [detector]
    sizes = [48]
    strides = [8]

    # Params :
    exp_name = sys.argv[5][:-3]
    params = importlib.import_module(exp_name)

    count = 0

    with open(list_file, 'r') as l_f:
        img_list = l_f.read().splitlines()
    s = ''
    for img_id in img_list:
        if count == 20000:
            break
        img_file = join(base_dir, img_id)
        img = cv2.imread(img_file)
        if max(img.shape) > 700:
            continue
        count += 1
        print img_file, img.shape,
        print count
        # Apply the detector with nms on the picture
        rois, scores = cascade(img, models, fprops,
                               params.local_scales,
                               sizes, strides,
                               params.overlap_ratio,
                               params.piece_size,
                               params.probs)
        # Transform into [x, y, w, h]

        for i in range(len(rois)):
            rois[i][1, :] = rois[i][1, :] - rois[i][0, :]
            e = [rois[i][0,0], rois[i][0,1], rois[i][1,0], rois[i][1,1]]
            s += img_id+' '+str(e[0])+' '+str(e[1])+' '+str(e[2])+' '+str(e[3])
            s += '\n'
    with open(out_file, 'w') as o_f:
        o_f.write(s)
    print 'Wrote at', out_file
