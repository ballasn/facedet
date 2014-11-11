import sys
import cPickle as pkl
from models.layer.convVariable import ConvElemwise as Cv_new_path
from utils.layer.convVariable import ConvElemwise as Cv_old_path
from models.layer.convVariable import ConvElemwise as Cv_new_path
from models.layer.corrVariable import CorrMMElemwise as Corr_new_path
from models.layer.cudnnVariable import CudNNElemwise as Cv_cudnn


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
    return [[x0, y0], [x1, y1]]




if __name__ == '__main__':
    m_f = '/data/lisatmp3/chassang/facedet/models/16/5layers16_best.pkl'
    with open(m_f, 'r') as mo:
        model = pkl.load(mo)
    print get_input_coords(1, 2, model)


