#! /Tmp/lisa/os_v3/canopy/bin/python

import cPickle as pkl
import sys
from facedet.models.layer.cudnnVariable import CudNNElemwise as CudLayer
from models.layer.cudnnVariable import CudNNElemwise as CudLayer2
from facedet.models.layer.convVariable import ConvElemwise as ConvLayer
from theano import config

config.floatX = 'float32'

def cudnnToConv(cudnn, conv_model):
    assert isinstance(cudnn, CudLayer) or isinstance(cudnn, CudLayer2)
    conv = ConvLayer(cudnn.output_channels,
                     cudnn.kernel_shape,
                     cudnn.layer_name,
                     cudnn.nonlinearity,
                     irange = 0.0,
                     kernel_stride=cudnn.kernel_stride,
                     max_kernel_norm=cudnn.max_kernel_norm,
                     tied_b=cudnn.tied_b,
                     pool_type=cudnn.pool_type,
                     pool_shape=cudnn.pool_shape,
                     pool_stride=cudnn.pool_stride)
    conv.mlp = conv_model
    conv.set_input_space(cudnn.input_space)

    conv.b.set_value(cudnn.b.get_value())
    conv.transformer._filters.set_value(cudnn.transformer._filters.get_value())
    return conv

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage %s: <model_cuddn.pkl> <model_conv.pkl>' % sys.argv[0]
        sys.exit(2)

    # Load both from file to avoid side effects
    with open(sys.argv[1], 'rb') as mo:
        cu_model = pkl.load(mo)
    print 'Loaded model'

    with open(sys.argv[1], 'rb') as mo:
        conv_model = pkl.load(mo)
    print 'Created new model'
    print type(conv_model)

    for l, layer in enumerate(cu_model.layers):
        if isinstance(layer, CudLayer) or isinstance(layer, CudLayer2):
            print 'Converting'
            conv_model.layers[l] = cudnnToConv(layer, conv_model)
            assert isinstance(conv_model.layers[l], ConvLayer)
    for layer in conv_model.layers:
        print layer
    with open(sys.argv[2], 'wb') as mout:
        pkl.dump(conv_model, mout)
    print 'Done'
