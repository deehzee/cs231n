import os

import cPickle as pkl
import keras
import numpy as np

"""
Utility functions to load CIFAR-10 data, split them into training,
validation and test data, and preprocess.
"""

def load_CIFAR_batch(filename):
    """Load single batch of CIFAR-10."""
    with open(filename, 'rb') as f:
        datadict = pkl.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
    """Load all of CIFAR-10."""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_{}'.format(b))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(rootdir,
                     num_training=45000,
                     num_validation=5000,
                     num_test=10000,
                     seed=None):
    """
    Load the CIFAR-10 dataset from disk, split into training,
    validation and test sets, and perform preprocessing to
    prepare it for classifiers.

    Inputs:
    - rootdir: The directory containing the CIFAR-10 data.
    - num_training: The size of the training split.
    - num_validation: The size of the validation split.
    - num_test: The size of the test split.
    - seed: Seed for numpy.random, for reproducibility.

    Returns the tuple:
    - data: A dictionary containing the preprocessed training,
      validation and test splits.
    - mean: The mean image.
    """

    # Load the data...
    X_train, y_train, X_test, y_test = load_CIFAR10(rootdir)

    # Split the data...
    if seed is not None:
        np.random.seed(seed)
    idxs = np.arange(50000)
    np.random.shuffle(idxs)
    train_mask = idxs[:num_training]
    val_mask = idxs[-num_validation:]
    test_mask = np.arange(num_test)
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Preprocess the data...
    # centering
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_val -= mean
    X_test -= mean
    # normalizing
    X_train /= 255
    X_val /= 255
    X_test /= 255

    # Package data into a dictionary
    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }

    return data, mean

