from pylearn2.config import yaml_parse
from pylearn2 import train
import math
import random
import os

# Defining the possible values of the layer parameters
NL_rng = [1]
num_channels_rng = [16, 32, 64, 128, 256, 512]
num_pieces_rng = [1, 4]
kernel_shape_rng = [3,8]
pool_shape_rng = [2,8]
pool_stride_rng = [1,2]
max_kernel_norm_rng = [0.9, 1.9, 2.9]

# Defining the possible values of the global parameters
batch_size_rng = [32, 64, 128, 256, 512]
lr_rng = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
max_epochs_rng = [10,100]
mom_rng = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

# Defining yalm template
yamlTemplate = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:datasets.faceDataset.faceDataset {
        which_set: 'train',
        positive_samples: "/data/lisatmp3/chassang/facedet/16/pos16_100_eq.npy",
        negative_samples: "/data/lisatmp3/chassang/facedet/16/neg16_100_eq.npy",
        axes: ['c', 0, 1, 'b']
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: %(Nb)d,
                layers: [
                %(layers)s
                 !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 2,
                     irange: .005
                 }
                ],
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [16, 16],
            num_channels: 3,
            axes: ['c', 0, 1, 'b'],
        },
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: %(LR)f,
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: %(mom)f,
        },
        train_iteration_mode: 'batchwise_shuffled_sequential',
        monitor_iteration_mode: 'even_sequential',
        monitoring_dataset:
            {
                'train' : *train,
                'valid' : !obj:datasets.faceDataset.faceDataset {
                        which_set: 'valid',
                        positive_samples: "/data/lisatmp3/chassang/facedet/16/pos16_100_eq.npy",
                        negative_samples: "/data/lisatmp3/chassang/facedet/16/neg16_100_eq.npy",
                        axes: ['c', 0, 1, 'b'],
                      },
            },
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: %(EP)d,
            new_epochs: True
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                channel_name: 'valid_y_misclass',
                save_path: %(outfile_ext)s
            }
    ],
    save_path: %(outfile)s,
    save_freq: 5
}
"""

# Sampling function
def sampleHyperparam(val_rng, stype="uniform"):
  if stype == "uniform":
    randValue = random.randint(val_rng[0], val_rng[1])
  elif stype == "log":
    logRandValue = random.randint(math.log(val_rng[0]), math.log(val_rng[1]))
    randValue = math.exp(logRandValue)
  elif stype == "discrete":
    idx = random.randint(0, len(val_rng)-1)
    randValue = val_rng[idx]
  return randValue


def getYamlForConvLayer(hyperparams, layer):
  
    template = """
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                 layer_name: %(name)s,
                 num_channels: %(NC)d,
                 kernel_shape: [%(KS)d, %(KS)d],
                 num_pieces: %(NP)d,
                 pool_shape: [%(Psh)d, %(Psh)d],
                 pool_stride: [%(Pst)d, %(Pst)d],
                 irange: .005,
                 max_kernel_norm: %(maxNorm)f,
                 tied_b: 1
                 },
    
    """
    
    yamlStr = template % {'name': layer,
                          'NC': hyperparams["num_channels"],
                          'KS': hyperparams["kernel_shape"],
                          'NP': hyperparams["num_pieces"],
                          'Psh': hyperparams["pool_shape"],
                          'Pst': hyperparams["pool_stride"],
                          'maxNorm': hyperparams["max_kernel_norm"]}

    return yamlStr
    
def getYamlFromHyperparams(hyperparams):
    
    # Get the name of the pickle file to which to save the model
    savefile = getFilename(hyperparams)
    
    # Get the yaml representation of the layers of the model
    layersYaml = ""
    
    for i in range(hyperparams['num_conv_layers']):
        layersYaml += getYamlForConvLayer(hyperparams, "h%i" % i )
        
    # Insert the global hyperparameters
    yamlContent = yamlTemplate % {'Nb' : hyperparams['batch_size'],
				  'layers' : layersYaml,
                                  'LR' : hyperparams['learning_rate'],
                                  'mom' : hyperparams['momentum'],
                                  'EP' : hyperparams['max_epochs'],
                                  'outfile_ext' : "./Results/" + savefile + "_best.pkl",
                                  'outfile' : "./Results/" + savefile + ".pkl"}
    return yamlContent
    

    
def getFilename(hyperparams):
  filename =  str(hyperparams['num_channels'])+"Ch_"+\
	      str(hyperparams['kernel_shape'])+"K_" +\
	      str(hyperparams['num_pieces'])+"Pi_" +\
	      str(hyperparams['pool_shape'])+"Psh"+str(hyperparams['pool_stride'])+"Pst_" +\
	      str(hyperparams['batch_size'])+"Nb_" +\
	      str(hyperparams['max_epochs'])+"EP_"+\
	      str(hyperparams['learning_rate'])+"LR_" +\
	      str(hyperparams['momentum'])+"mom"
  return filename
  

if __name__ == '__main__':
    
    trials = 1
    
    for i in range(trials):
      
      print "Sampling hyperparameters trial "+str(i)+" out of "+str(trials)

      # Sample hyperparameters
      hyperparams = {        
        # Sample global parameters
        'num_conv_layers' : sampleHyperparam(NL_rng,"discrete"),
        'learning_rate' : sampleHyperparam(lr_rng,"discrete"),
        'momentum' : sampleHyperparam(mom_rng,"discrete"),
        'max_epochs' : sampleHyperparam(max_epochs_rng,"uniform"),
        'batch_size' : sampleHyperparam(batch_size_rng,"discrete"),
        
	# Sample layer 1 parameters
	'num_channels' : sampleHyperparam(num_channels_rng,"discrete"),
	'kernel_shape' : sampleHyperparam(kernel_shape_rng,"uniform"),
	'num_pieces' : sampleHyperparam(num_pieces_rng,"uniform"),
	'pool_shape' : sampleHyperparam(pool_shape_rng,"uniform"),
	'pool_stride' : sampleHyperparam(pool_stride_rng,"uniform"),
	'max_kernel_norm' : sampleHyperparam(max_kernel_norm_rng,"discrete")
      }

      # Output the yaml file only if it was never generated before
      print "Generating yalm file"

      savepath = "./Configurations/"
      if not os.path.exists(savepath):
        os.makedirs(savepath)
        
      filename = savepath + getFilename(hyperparams) + ".yaml"
      if not os.path.exists(filename):
	yamlContent = getYamlFromHyperparams(hyperparams)
	f = open(filename, "w")
        f.write(yamlContent)
        f.close()
        
      print "Loading model"
      with open(filename, "r") as fp:
	model = yaml_parse.load(fp)

      # Train
      print "Training model"
      model.main_loop()

