import cPickle as pkl
from facedet.models.layer.convVariable import ConvElemwise as Cv_new_path
from models.layer.convVariable import ConvElemwise as Cv_f
from facedet.utils.layer.convVariable import ConvElemwise as Cv_old_path
from facedet.models.layer.corrVariable import CorrMMElemwise as Corr_new_path
from facedet.models.layer.cudnnVariable import CudNNElemwise as Cv_cudnn


def get_input_coords(i, j, model):
    """
    Returns the coords in the original input given the ones
    at the output
    -----------------------------
    model : model including conv layers
    """
    x0 = i
    y0 = j
    for l in model.layers[::-1]:
        if isinstance(l, Cv_new_path) or isinstance(l, Cv_old_path) or\
           isinstance(l, Corr_new_path) or isinstance(l, Cv_cudnn) or\
           isinstance(l, Cv_f):
            if l.pool_type is not None:
                x0 *= l.pool_stride[0]
                y0 *= l.pool_stride[0]

            x0 *= l.kernel_stride[0]
            y0 *= l.kernel_stride[0]
    return [[x0, y0],
            [model.layers[0].input_space.shape[0],
             model.layers[0].input_space.shape[1]]]


if __name__ == '__main__':
    m_f = '../models/modele96_conv.pkl'
    m_f = '../cascade/models/dsn48large_sig_best_fine_unfreeze.pkl'
    with open(m_f, 'r') as mo:
        model = pkl.load(mo)
    print get_input_coords(1, 0, model)
    print get_input_coords(0, 1, model)
    print get_input_coords(0, 0, model)
    print model.layers[0].input_space.shape[0]
