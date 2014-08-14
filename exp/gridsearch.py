from pylearn2.config import yaml_parse
from pylearn2 import train
from math import log10



# Defining the range of the param
param_range = [25, 30, 35, 45, 50]

# Looping over values
for val in param_range:

    # Reading the original yaml
    with open("simple_net.yaml", "r") as fp:
        model = yaml_parse.load(fp)

    print model.algorithm.termination_criterion
    model.algorithm.termination_criterion._max_epochs = 150
    print model.algorithm.termination_criterion._max_epochs

    # Modify value of param
    model.model.layers[0].dim = val

    # Modify the save path accordingly
    dim = model.model.layers[0].dim
    print type(model.algorithm.learning_rate.get_value())
    log_LR = int(log10(model.algorithm.learning_rate.get_value()))
    mom = model.algorithm.learning_rule.momentum.get_value()
    mom = round(mom, 2)

    # Defining save_path
    # <dim>d<log_LR>LR<momentum>mom.pkl
    model.save_path = "./100k_simple/" +\
        str(dim)+'d'+str(log_LR)+'LR'+str(mom)+"mom.pkl"
    print model.save_path

    # Train
    model.main_loop()

print "-"*11
print "- THE END -"
print "-"*11
