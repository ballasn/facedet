from pylearn2.config import yaml_parse
from pylearn2 import train
import math
import random
import os
import sys
import getopt
import numpy as np
import cPickle as pkl
from utils.compression.TeacherHintRegressionCost import TeacherHintRegressionCost
from models.layer.convVariable import ConvElemwise
from models.layer.SoftmaxBC01Extended import SoftmaxExtended
from models.layer.SigmoidBC01Extended import SigmoidExtended
from pylearn2.models.mlp import Sigmoid, Softmax, RectifiedLinear, ConvRectifiedLinear, RectifierConvNonlinearity, SigmoidConvNonlinearity, TanhConvNonlinearity
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.monitor import push_monitor
from pylearn2.space import VectorSpace
from copy import deepcopy
from models.layer.convVariable import ConvElemwise
from pylearn2.models.mlp import IdentityConvNonlinearity

from pylearn2.train import Train


def main(argv, freeze):

  try:
    opts, args = getopt.getopt(argv, '')
    yaml = args[0]
    model = args[1]
  except getopt.GetoptError:
    usage()
    sys.exit(2)

  # Load yaml
  with open(yaml, "r") as sty:
    train = yaml_parse.load(sty)


  # Load pretrained model with bad sigmoid output
  with  open(model, 'r') as fo:
    model = pkl.load(fo)

  # Remove the last layer, puts a real sigmoid instead
  if freeze:
    for i in range(0, len(model.layers) - 2):
      model.freeze(model.layers[i].get_params())


  ### Add last conv elemwise
  layer = ConvElemwise(layer_name= 'out',
                       output_channels= 1,
                       kernel_shape=[2,2],
                       irange=0.05,
                       nonlinearity=IdentityConvNonlinearity(),
                       max_kernel_norm= 7.9,
                       tied_b=1)
  layer.set_mlp(model)
  layer.set_input_space(model.layers[-3].get_output_space())
  model.layers[-2] = layer

  ### Add Sigmoid
  layer = SigmoidExtended(layer_name='y', n_classes=1)
  layer.set_mlp(model)
  layer.set_input_space(model.layers[-2].get_output_space())
  model.layers[-1] = layer

  #print model.layers
  #model.monitor = train.model.monitor
  #train.model = model
  train.model = push_monitor(model, "old")
  print train.model


  #train = Train(train.dataset, model, train.algorithm, train.save_path,
  #                train.save_freq, train.extensions, train.allow_overwrite)
  train.main_loop()


if __name__ == "__main__":
  main(sys.argv[1:], freeze=True)
