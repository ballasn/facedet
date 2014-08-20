from pylearn2.config import yaml_parse
from pylearn2 import train
from math import log10



# Defining the range of the param
param_range = [0.0000005,0.0000001]

# Looping over values
for val in param_range:

    # Reading the original yaml
    with open("simple_net.yaml", "r") as fp:
        model = yaml_parse.load(fp)
    print "Loaded model"

    # Define test length
    model.algorithm.termination_criterion._max_epochs = 300

    # Modify value of param
    model.algorithm.learning_rate.set_value(val)

    # We need to recreate the layers now
    print "redifining layers"
    model.model.layers[0].set_input_space(model.model.input_space)
    model.model.layers[1].set_input_space(model.model.layers[0].output_space)

    # Modify the save path accordingly
    dim = model.model.layers[1].dim
    print type(model.algorithm.learning_rate.get_value())
    log_LR = int(log10(model.algorithm.learning_rate.get_value()))
    mom = model.algorithm.learning_rule.momentum.get_value()
    mom = round(mom, 2)
    k_shape = model.model.layers[0].kernel_shape[0]
    irange = model.model.layers[1].irange

    # Defining save_path
    # <dim>d<log_LR>LR<momentum>mom.pkl
    model.save_path = "./100k_simple/" +\
            str(dim)+'d'+str(k_shape)+"k"+str(log_LR)+"lr.pkl"
    print model.save_path
    model.extensions[0].save_path ="./100k_simple/" +\
            str(dim)+'d'+str(k_shape)+"k"+str(log_LR)+"lr_best.pkl"

    # Train
    model.main_loop()

print "-"*11
print "- THE END -"
print "-"*11
