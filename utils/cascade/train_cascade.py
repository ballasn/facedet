from pylearn2.config import yaml_parse
from theano import function
import theano.tensor as T
import numpy as np
import cPickle as pkl
from remove_hdf5 import remove_inactives


try:  # Nicolas config
    from facedet.datasets.faceDataset_hdf5 import faceDataset
except ImportError:  # Antoine's config
    from datasets.faceDataset_hdf5 import faceDataset


def train_one_stage(model_yaml, dataset=None, model_file=None,
                    max_epochs=500):
    """
    Train a stage of the cascade
    Return the path to the best model.
    ----------------------------------
    model_yaml : YAML file defining the model
    dataset : dataset object to train on
    model_file : target of save
    max_epochs : number of training epochs
    """
    with open(model_yaml, "r") as m_y:
        model = yaml_parse.load(m_y)
    print "Loaded YAML"
    #############################
    model.algorithm.termination_criterion._max_epochs = max_epochs
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


def define_inactives(model_file, dataset, acc_prob, batch_size=128,
                     nb_classes=2):
    """
    Return the indices of uninteresting negatives,
        eg examples that have a highly negative response
    ----------------------------------------------
    model_file : pkl file containing the classifier
    dataset : dataset to be filtered
    acc_prob : threshold to consider an ex as inactive
            if p(face) < acc_prob then the ex is considered inactive
    """
    with open(model_file, 'r') as m_f:
        model = pkl.load(m_f)
    # Compile network prediction function
    x = T.tensor4("x")
    predict = function([x], model.fprop(x))

    # Compute prediction of training set
    print "Defining inactives"
    batches = dataset.iterator(mode='negative_seq',
                               batch_size=batch_size)
    cur = 0
    inactives = []
    for x, y in batches:
        t = np.reshape(x, ((batch_size,) + tuple(dataset.img_shape)))
        # Transform into C01B format
        t = np.transpose(t, (3, 1, 2, 0))
        preds = predict(t)
        preds = np.reshape(preds, (batch_size, nb_classes))
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
    positives = "/data/lisatmp3/chassang/facedet/16/pos16.hdf"
    negatives = "/data/lisatmp3/chassang/facedet/16/neg16.hdf"
    hdf_dir = "/data/lisatmp3/chassang/facedet/16/"
    ratio = 0.8
    acc_prob = 0.05
    batch_size = 128
    m_yaml = "../../exp/simple_net_extended.yaml"
    dataset = faceDataset(positives, negatives, 'train')
    old_neg_file = negatives

    for i in range(20):
        # Train one stage
        m_file = train_one_stage(m_yaml, dataset=dataset,
                                 model_file="cascadeTest"+str(i))

        # Get the list of uninteresting negatives
        #m_file = 'cascadeTest0'
        inactives = define_inactives(m_file+"_best.pkl", dataset, acc_prob)
        #inactives = np.load('inactives'+str(0)+".npy")
        inactives = np.array(inactives)
        inac_dir = "inactives"+str(i)+".npy"

        np.save(inac_dir, inactives)

        # Define the new file
        new_neg_file = hdf_dir + "neg_" + str(i) + ".hdf"
        print new_neg_file
        inactives = np.unique(inactives)
        remove_inactives(old_neg_file, new_neg_file, inactives)

        # Define the new dataset without the inactives
        dataset = faceDataset(positives, new_neg_file, 'train')
        old_neg_file = new_neg_file



