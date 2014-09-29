from pylearn2.config import yaml_parse
from theano import function
import theano.tensor as T
import numpy as np
import cPickle as pkl
from utils.create_dataset.lists_to_hdf import hdf_from_textfile


try:  # Nicolas config
    from facedet.datasets.faceDataset_hdf5 import faceDataset
except ImportError:  # Antoine's config
    from datasets.faceDataset_hdf5 import faceDataset


def train_one_stage(model_yaml, dataset=None, max_epochs=500):
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

    # Define the length of training
    model.algorithm.termination_criterion._max_epochs = max_epochs

    if dataset is not None:
        # We use the specified dataset
        model.dataset = dataset

    # Train model
    model.main_loop()

    # Return the path to the best model obtained
    best_model_file = model.extensions[0].save_path
    return best_model_file


def define_inactives(model_file, dataset,
                     input_text,
                     output_text,
                     nb_examples,
                     acc_prob, batch_size=128,
                     nb_classes=2):
    """
    Write a text file containing the negatives that would have passed
    the previous stage of the stage
    ----------------------------------------------
    model_file : pkl file containing the classifier
    dataset : dataset which contains examples in the same order as in
              input_file
    input_text : file containing the coords of patches in dataset
    output_text : file in which we'll write results
    nb_examples : number of examples considered in the input file
    acc_prob : threshold to consider an ex as inactive
            if p(face) < acc_prob then the ex is considered inactive
    """

    # Define network prediction function
    with open(model_file, 'r') as m_f:
        model = pkl.load(m_f)
    x = T.tensor4("x")
    predict = function([x], model.fprop(x))

    # Compute prediction of training set
    print "Defining inactives"
    batches = dataset.iterator(batch_size=batch_size)

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
            # Look for the negatives such that
            # 0 : p(face)    1 : p(non-face)
            if preds[i, 0] < acc_prob:
                    inactives.append(cur)
            cur += 1

    # Now write the results at output_text
    with open(input_text, 'r') as in_text:
        with open(output_text, 'w') as out_text:
            in_list = in_text.readlines()
            in_list = in_list[:nb_examples]
            out_list = []
            for e in inactives:
                in_list[e] = None
            for e in in_list:
                if e is not None:
                    out_list.append(e)
            for l in out_list:
                out_text.write(l)

    # Returns the number of examples in the new list
    return len(out_list)


def make_new_dataset():
    """
    make a new dataset from the lists
    """

    return None


if __name__ == '__main__':

    # Define parameters
    acc_prob = 0.5
    batch_size = 128
    nb = 10000

    # Files to be used for training
    list_pos = "../create_dataset/text_files/pos700_shuffled.txt"
    list_neg = "../create_dataset/text_files/neg700_shuffled.txt"
    input_text = list_neg
    base_hdf = '/data/lisatmp3/chassang/facedet/'

    hdf_pos16 = base_hdf + '16/pos700_new.hdf'
    hdf_pos48 = base_hdf + '48/pos700_new.hdf'
    hdf_pos96 = base_hdf + '96/pos700_new.hdf'

    yaml16 = '../../exp/convtest/large16.yaml'
    yaml48 = '../../exp/convtest/large48.yaml'
    yaml96 = '../../exp/convtest/large96.yaml'

    new_hdf_neg = base_hdf + '16/neg700_new.hdf'
    # Define arrays
    yamls = [yaml16, yaml48, yaml96]
    hdf_pos = [hdf_pos16, hdf_pos48, hdf_pos96]
    sizes = [16, 48, 96]
    assert len(sizes) == len(yamls)

    for i in range(len(yamls)):
        # Define the dataset
        dataset = faceDataset(hdf_pos[i], new_hdf_neg,
                              'train', nb_examples=[nb, nb], ratio=1.0)

        # Train one stage
        best_model_file = train_one_stage(yamls[i])

        # Write the new list of negative examples
        output_text = list_neg[:-4] + str(i) + '.txt'
        nb = define_inactives(best_model_file, dataset,
                              input_text, output_text,
                              acc_prob)
        input_text = output_text
        if i == len(yamls)-1:
            break

        # Create the new HDF5 for negative examples
        new_hdf_neg = '/data/lisatmp3/chassang/facedet/'+str(sizes[i+1]) +\
                      '/neg_new.hdf'
        hdf_from_textfile(output_text, sizes[i+1], new_hdf_neg)

    print 'Done with the cascade'
