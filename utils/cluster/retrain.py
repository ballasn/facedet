import sys
import os.path
import cPickle as pkl
from pylearn2.monitor import push_monitor
from pylearn2.utils import serial


def train_again(yaml):
    '''
    Relaunch training of a model with conditions specified by the YAML
    Looks for the model file defined by save path and replace the model
    instanciated by the one that was trained before
    -------------------------------------------------------------------
    yaml : string, filename
           YAML file defining the exp to be continued
    '''

    context = serial.load_train_file(yaml)
    print "\tLoaded YAML"

    # Load the trained model
    model_file = context.save_path

    if not os.path.isfile(model_file):
        model_file = context.extensions[-1].save_path

    with open(model_file, 'r') as m_f:
        trained_model = pkl.load(m_f)

    # Define the continuing one
    new_model = push_monitor(trained_model, 'trained_model')
    # Define it as the model to be trained
    context.model = new_model
    # Train again
    context.main_loop()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage : %s <yaml_file>' % sys.argv[0]
        sys.exit(2)
    train_again(sys.argv[1])
