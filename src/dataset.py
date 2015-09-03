#!/usr/bin/env python2


""" Contains functions for working with datasets.
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from array import array


def load_mnist_all(filename_train_data, filename_train_labels, filename_test_data, filename_test_labels):
    """ Loads the data from the MNIST dataset. Expects the full path of the
        compressed files.
    """
    train = load_mnist(filename_train_data, filename_train_labels)
    test = load_mnist(filename_test_data, filename_test_labels) 
    return (train, test)


def load_mnist(filename_images, filename_labels):
    """ Loads the data file and labels.
        Adapted from: https://github.com/sorki/python-mnist.
    """
    # Load the images
    with open(filename_images, 'rb') as f:
        (magic, size, rows, cols) = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError('magic number mismatch: expected 2051 but got %d' % magic)
        image_data = array("B", f.read())

    # Load the labels
    with open(filename_labels, 'rb') as f:
        (magic, size) = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError('magic number mismatch: expected 2049 but got %d' % magic)
        Y = array("B", f.read())

    X = []
    for i in xrange(size):
        X.append([0]*rows*cols)
    for i in xrange(size):
        X[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

    return (X, Y)


def viz_categorical(X, Y, classes):
    """ Displays a number of training instances from each class.
    """
    K = len(classes)
    samples_per_class = 7
    for (y, cls) in enumerate(classes):
        idxs = np.flatnonzero(Y == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for (i, idx) in enumerate(idxs):
            plt_idx = i*K + y + 1
            plt.subplot(samples_per_class, K, plt_idx)
            image = np.reshape(X[idx], (28, 28))
            plt.imshow(image.astype('uint8'), interpolation='none', cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


def test_mnist():
    filenames = ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
                 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte']
    filenames = ['../datasets/MNIST/' + i for i in filenames]
    (train, test) = load_mnist_all(filenames[0], filenames[1], filenames[2],
                               filenames[3])

    # Display a few examples from each class to make sure we got this working
    X = np.array(train[0])
    Y = np.array(train[1])
    classes = [str(i) for i in range(10)]
    viz_categorical(X, Y, classes)


def main():
    import pdb
    test_mnist()


if __name__ == '__main__':
    main()
 
