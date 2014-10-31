from pylearn2.config import yaml_parse
from pylearn2 import train
import math
import random
import os
import sys
import getopt
import numpy as np
import cPickle as pkl


from pylearn2.train import Train


def main(argv, freeze):

  try:
    opts, args = getopt.getopt(argv, '')
    modelfile = args[0]
  except getopt.GetoptError:
    usage()
    sys.exit(2)

  # Load pretrained model with bad sigmoid output
  with  open(modelfile, 'r') as fo:
    model = pkl.load(fo)

  print model.freeze_set
  model.freeze_set.clear()
  print model.freeze_set


  with open(modelfile[:-4] + "_unfreeze.pkl", "wb") as fo:
    pkl.dump(model, fo)


if __name__ == "__main__":
  main(sys.argv[1:], freeze=True)
