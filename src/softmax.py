#!/usr/bin/env python2

""" Implements a multiclass linear softmax + cross-entropy classifier.
"""


import numpy as np


class Softmax(LinearClassifier):
    """ Encapsulates a multiclass linear softmax + cross-entropy classifier.
    """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss(self.W, X_batch, y_batch, reg)


def softmax_loss(W, X, y, reg):
    """ Softmax loss function (vectorized version).
        @param W    C x D array of weights
        @param X    D x N array of data
        @param y    N length array of training labels with values 0, 1, ..., K-1
                    for K classes
        @param reg  Regularization strength
        @return     2-tuple of the loss of the current parameterization and the
                    gradient with respect to weights W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    K = W.shape[0]
    N = y.shape[0]

    # Compute scores
    f = np.dot(W, X)

    # Shift f for numerical stability
    f -= np.max(f, axis=0)

    # Compute the probabilities
    ef = np.exp(f)
    p = ef/np.sum(ef, axis=0)
    py = p[y, xrange(N)]

    # Compute the loss
    loss = -np.sum(np.log(py))/N + reg*np.sum(W*W)

    # Compute the gradient
    p[y, xrange(N)] -= 1
    dW = np.dot(p, np.transpose(X))/N + 2*reg*W

    return loss, dW
