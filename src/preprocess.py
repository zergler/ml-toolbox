#!/usr/bin/env python2

""" Implements functions for preprocessing data.
"""


def subtract_mean(X):
    """ Subtracts the mean of the data from each training point.

        @param X    Data to compute and subtract mean from
        @return     Data with mean subtracted
    """
    mean = np.mean(X, axis=1)
    return X - mean_image


def split_train_val(X, y, ratio=0.6):
    """ Forms a training and validation set from a set of data using the given
        ratio.

        @param X        D x N array of data
        @param ratio    Determines the number of training points (float)

        Note: make sure y is a 1-tuple because code might not work otherwise.
    """
    assert(len(y.shape) == 1)
    split = int(X.shape[1]*ratio)
    X_train = X[:, 0:split]
    y_train = y[0:split]
    X_val = X[:, split:]
    y_val = y[split:]
    train = (X_train, y_train)
    val = (X_val, y_val)
    return (train, val)
