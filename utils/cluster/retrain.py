import sys
import cPickle as pkl
from pylearn2.config import yaml_parse
from pylearn2.monitor import push_monitor


def train_again(yaml):
    '''
    Relaunch training of a model with conditions specified by the YAML
    Looks for the model file defined by save path and replace the model
    instanciated by the one that was trained before
    -------------------------------------------------------------------
    yaml : string, filename
           YAML file defining the exp to be continued
    '''
    with open(yaml, "r") as m_y:
        context = yaml_parse.load(m_y)
    print "\tLoaded YAML"

    # Load the trained model
    model_file = context.save_path
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
