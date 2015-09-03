#!/usr/bin/env python2

""" Implements a multiclass linear svm.
"""


import numpy as np
from linear_classifier import LinearClassifier


class LinSVM(LinearClassifier):
    """ Encapsulates a multiclass linear SVM.
    """
    def loss(self, X_batch, y_batch, reg):
        return svm_loss(self.W, X_batch, y_batch, reg)


def svm_loss(W, X, y, reg):
    """ Structured SVM loss function (vectorized implementation).

        @param W    K x D array of weights
        @param X    D x N array of data
        @param y    N length array of training labels with values 0, 1, ..., K-1
                    for K classes
        @param reg  Regularization strength
        @return     2-tuple of the loss of the current parameterization and the
                    gradient with respect to weights W
    """
    loss = 0.0
    dW = np.zeros_like(W)
    delta = 1
    N = y.shape[0]

    # Compute scores
    f = np.dot(W, X)
    fy = f[y, xrange(N)]

    # Compute the margins
    margin = f - fy + delta 

    # Set all ground truth margins to zero (otherwise they would have delta term)
    margin[y, xrange(N)] = 0

    # Compute the total loss
    loss = np.sum(np.maximum(0, margin))/N + reg*np.sum(W*W)

    # Compute the gradient
    S = np.array(margin > 0).astype(int)
    S[y, xrange(N)] = -np.sum(S, axis=0)
    dW = np.dot(S, np.transpose(X))/N + 2*reg*W

    return loss, dW

