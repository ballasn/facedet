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


def train_one_stage(model_yaml, dataset=None):
    """
    Train a stage of the cascade
    Return the path to the best model.
    """
    with open(model_yaml, "r") as m_y:
        model = yaml_parse.load(m_y)
    print "Loaded first YAML"
    #############################
    model.algorithm.termination_criterion._max_epochs = 200
    #############################
    # Write info about the model
    model_file = model.extensions[0].save_path
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

    positives = "/data/lisatmp3/chassang/facedet/16/pos16_100_eq.npy"
    negatives = "/data/lisatmp3/chassang/facedet/16/neg16_100_eq.npy"
    ratio = 0.8
    # which_set = 'test' return all examples ???
    dataset = faceDataset(positives, negatives, 'coucou')
    batch_size = 128
    batches = dataset.iterator(mode='negative_seq',
                               batch_size=batch_siize)
    # We skip the first training for convenience
    m_yaml = "../exp/simple_net_multidim.yaml"
    model_file = '../exp/sn_1kiter_best.pkl'

    # Get the list ofi uninteresting negatives
    inactives = define_inactives(model_file, dataset, 0.05)
    print 0.05, len(inactives)
    inactives = np.array(inactives)
    np.save("inactives.npy", inactives)

    # Define the new dataset without the inactives
    dataset_bis = faceDataset(positives, negatives, 'train',
            inactive_examples="inactives.npy")

    # Train a second stage on the modified dataset
    m_file_bis = train_one_stage(m_yaml, dataset_bis)

    inactives2 = define_inactives(predict, dataset, 0.05)
    print 0.05, len(inactives)
    print 0.05, len(inactives2)



