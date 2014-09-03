from pylearn2.config import yaml_parse
from pylearn2 import train
import math
import random
import os
import numpy as np

# Defining the possible values of the layer parameters
range1 = lambda start, end: range(start, end+1)

inputDim = [16, 16]
NL_rng = range1(1,4)
num_channels_rng = [16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304]
num_pieces_rng = range1(1,4)
kernel_shape_rng = range1(3,8)
pool_shape_rng = range1(2,8)
pool_stride_rng = range1(1,2)
max_kernel_norm_rng = [0.9, 1.9, 2.9]

# Defining the possible values of the global parameters
batch_size_rng = [32, 64, 128, 256, 512]
lr_rng = [0.000001, 1]
max_epochs_rng = [10,10000]
mom_rng = [0.1, 1.0]

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
            shape: [%(inDim_x)d, %(inDim_y)d],
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
  elif stype == "loguniform":
    logRandValue = random.uniform(math.log(val_rng[0]), math.log(val_rng[1]))
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
                          'NC': hyperparams[layer+"_num_channels"],
                          'KS': hyperparams[layer+"_kernel_shape"],
                          'NP': hyperparams[layer+"_num_pieces"],
                          'Psh': hyperparams[layer+"_pool_shape"],
                          'Pst': hyperparams[layer+"_pool_stride"],
                          'maxNorm': hyperparams[layer+"_max_kernel_norm"]}

    return yamlStr
    
def getYamlFromHyperparams(hyperparams):
    
    # Get the name of the pickle file to which to save the model
    savefile = getFilename(hyperparams)
    
    # Get the yaml representation of the layers of the model
    layersYaml = ""
    
    for i in np.arange(hyperparams['num_CL'])+1:
        layersYaml += getYamlForConvLayer(hyperparams, "C%i" % i )
        
    # Insert the global hyperparameters
    yamlContent = yamlTemplate % {'Nb' : hyperparams['batch_size'],
				  'layers' : layersYaml,
				  'inDim_x': inputDim[0],
				  'inDim_y': inputDim[1],
                                  'LR' : hyperparams['learning_rate'],
                                  'mom' : hyperparams['momentum'],
                                  'EP' : hyperparams['max_epochs'],
                                  'outfile_ext' : "./RandSearch/Deep_Results/" + savefile + "_best.pkl",
                                  'outfile' : "./RandSearch/Deep_Results/" + savefile + ".pkl"}
    return yamlContent
    

    
def getFilename(hyperparams):
  # Global parameters
  filename =  str(hyperparams['num_CL'])+"CL_" +\
	      str(hyperparams['batch_size'])+"Nb" +\
	      str(hyperparams['max_epochs'])+"EP"+\
	      str(hyperparams['learning_rate'])+"LR" +\
	      str(hyperparams['momentum'])+"mom_"
  
  # Layer parameters
  for i in np.arange(hyperparams['num_CL'])+1:
    filename += str("_C%i" % i) +\
		str(hyperparams["C%i_" % i+'num_channels'])+"Ch"+\
		str(hyperparams["C%i_" % i+'kernel_shape'])+"K" +\
		str(hyperparams["C%i_" % i+'num_pieces'])+"Pi" +\
		str(hyperparams["C%i_" % i+'pool_shape'])+"Psh"+\
		str(hyperparams["C%i_" % i+'pool_stride'])+"Pst"
  return filename
  

if __name__ == '__main__':
    
    trials = 1
    
    for i in range(trials):
      
      layerIn = np.array(inputDim)
      maxKernel = min(layerIn)
      
      print "Sampling hyperparameters trial "+str(i)+" out of "+str(trials)

      # Sample hyperparameters
      hyperparams = {        
        # Sample global parameters 
        'num_CL' : sampleHyperparam(NL_rng,"discrete"),
        'learning_rate' : sampleHyperparam(lr_rng,"loguniform"),
        'momentum' : sampleHyperparam(mom_rng,"loguniform"),
        'max_epochs' : sampleHyperparam(max_epochs_rng,"loguniform"),
        'batch_size' : sampleHyperparam(batch_size_rng,"discrete")
      }  
            
      # Sample convolutional layer parameters
      for l in np.arange(hyperparams['num_CL'])+1:
	kernel_shape_rng = [min(kernel_shape_rng[0],maxKernel),min(kernel_shape_rng[1],maxKernel)]

	name = 'C%i'% l
	layerparams = {
	  name +'_num_channels' : sampleHyperparam(num_channels_rng,"discrete"),
	  name +'_kernel_shape' : sampleHyperparam(kernel_shape_rng,"discrete"),
	}
	
	detect = [int(layerIn[0]-layerparams[name +'_kernel_shape']+1),int(layerIn[1]-layerparams[name +'_kernel_shape']+1)]
	pool_shape_rng = range1(min(min(pool_shape_rng),detect[0]),\
			  min(max(pool_shape_rng),detect[1]))	  	

	layerparams.update({
	  name +'_num_pieces' : sampleHyperparam(num_pieces_rng,"discrete"),
	  name +'_pool_shape' : sampleHyperparam(pool_shape_rng,"discrete")
	})
	
	pool_stride_rng = range1(min(min(pool_shape_rng), layerparams[name+'_pool_shape']),min(max(pool_shape_rng), layerparams[name+'_pool_shape']))
	
	layerparams.update({
	  name +'_pool_stride' : sampleHyperparam(pool_stride_rng,"discrete"),
	  name +'_max_kernel_norm' : sampleHyperparam(max_kernel_norm_rng,"discrete")
	})
	hyperparams.update(layerparams)
	layerIn = np.ceil((np.array(detect)-layerparams[name +'_pool_shape'])/float(layerparams[name +'_pool_stride']))+1

	maxKernel = min(min(layerIn),layerparams[name +'_kernel_shape'])

	
      # Output the yaml file only if it was never generated before
      print "Generating yalm file"

      savepath = "./RandSearch/Deep_Configurations/"
      if not os.path.exists(savepath):
        os.makedirs(savepath)
        
      filename = savepath + getFilename(hyperparams) + ".yaml"
      if not os.path.exists(filename):
	yamlContent = getYamlFromHyperparams(hyperparams)
	f = open(filename, "w")
        f.write(yamlContent)
        f.close()
        
      #try:    
	print "Loading model"
	with open(filename, "r") as fp:
	  model = yaml_parse.load(fp)

	# Train
	print "Training model"
	model.main_loop()
      #except:
	#print "Trial "+str(i)+" failed."
