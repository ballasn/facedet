from pylearn2.config import yaml_parse
from theano import function
import theano.tensor as T
from theano import config
import numpy as np
import cPickle as pkl


try:  # Nicolas config
    from facedet.datasets.faceDataset_cascade import faceDataset
except ImportError:  # Antoine's config
    from datasets.faceDataset_cascade import faceDataset


def train_one_stage(model_yaml, dataset=None, model_file=None):
    """
    Train a stage of the cascade
    Return the path to the best model.
    """
    with open(model_yaml, "r") as m_y:
        model = yaml_parse.load(m_y)
    print "Loaded YAML"
    #############################
    model.algorithm.termination_criterion._max_epochs = 1000
    #############################
    # Write info about the model
    if model_file is None:
        model_file = model.extensions[0].save_path
    else:
        model.extensions[0].save_path = model_file + "_best.pkl"
        model.save_path = model_file + ".pkl"

    if dataset is not None:
        # We use the modified dataset
        model.dataset = dataset

    # Train model
    model.main_loop()

    return model_file


def define_inactives(model_file, dataset, acc_prob):
    """
    Return the indices of inactives
    Filter the dataset
    """
    with open(model_file, 'r') as m_f:
        model = pkl.load(m_f)
    # Compile network prediction function
    x = T.tensor4("x")
    predict = function([x], model.fprop(x))
    # Compute prediction of training set
    print "Defining inactives"
    batch_size = 64
    nb_classes = 2
    preds = np.zeros((batch_size, nb_classes),
                     dtype=config.floatX)
    batches = dataset.iterator(mode='negative_seq',
                               batch_size=batch_size)
    cur = 0
    inactives = []
    for x, y in batches:
        t = np.reshape(x, ((batch_size,) + tuple(dataset.img_shape)))
        # Transform into C01B format
        t = np.transpose(t, (3, 1, 2, 0))
        preds = predict(t)
        # Loop over examples
        for i in range(preds.shape[0]):
            # Look for the negatives which
            # are the most easily classified
            # 0 : p(face)    1 : p(non-face)
            if preds[i, 0] < acc_prob:
                    inactives.append(cur)
            cur += 1

    return inactives


if __name__ == '__main__':

    # Define parameters
    positives = "/data/lisatmp3/chassang/facedet/16/pos16_700_eq.npy"
    negatives = "/data/lisatmp3/chassang/facedet/16/neg16_700_eq.npy"
    ratio = 0.8
    acc_prob = 0.1
    # which_set = 'test' return all examples ???
    dataset = faceDataset(positives, negatives, 'train')
    batch_size = 128
    m_yaml = "../exp/simple_net_extended.yaml"
    for i in range(3):
        # Train one stage
        m_file = train_one_stage(m_yaml, dataset=dataset,
                                 model_file="cascade"+str(i))

        # Get the list of uninteresting negatives
        inactives = define_inactives(m_file+"_best.pkl", dataset, acc_prob)
        print 0.05, len(inactives)
        inactives = np.array(inactives)
        np.save("inactives"+str(i)+".npy", inactives)
        if i != 0:
            past_inactives = np.load("inactives"+str(i-1)+".npy")
            new_inactives = np.concatenate((past_inactives, inactives), axis=0)
            new_inactives = np.unique(new_inactives)
            np.save("inactives"+str(i)+".npy", new_inactives)

        # Define the new dataset without the inactives
        dataset = faceDataset(positives, negatives, 'train',
                        inactive_examples="inactives"+str(i)+".npy")




