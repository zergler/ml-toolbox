
""" Implements a linear classifier base class which can be inherited by linear
    models.
"""

import random

import numpy as np
import matplotlib.pyplot as plt


class LinearClassifier:
    """ Encapsulates a linear classifier.
    """

    def __init__(self, learning_method):
        """ Initializes the linear classifier.
            
        """
        self.W = None

    def init_weights_std_normal(self, K, D, sigma):
        """ Initializes the weights using a standard normal .
        """
        # Randomly initialize weights
        if mode == 'random':
            self.W = np.random.randn(K, D)*0.001

    def train_sgd(self, X, y, eta=1e-3, reg=1e-5, epochs=100, s=200, verbose=False):
        """ Trains the linear classifier using stochastic gradient descent

            @param X        D x N array of training data
            @param y        N length array of training labels with values 0, 1,
                            ..., K-1 for K classes
            @param eta      Learning rate for optimization
            @param reg      Regularization term of loss
            @param epochs   Number of iterations to use for optimization
            @param s        Batch size to use for each epoch of training
            @param verbose  Prints out the progress of learning if true
            @return         List containing the loss at each epoch
        """
        (D, N) = X.shape
        K = np.max(y) + 1
        if self.W is None:
            init_weights_std_normal(K, D, 0.001)

        # Give a reasonable step size for displaying epochs
        verb_steps = epochs/10

        losses = []
        pop = range(num_train)
        for it in xrange(epochs):
            # Grab some batches
            indices = random.sample(pop, batch_size)
            X_batch = X[:, indices]
            y_batch = y[indices]

            # Evaluate loss and gradient
            (L, G) = self.loss(X_batch, y_batch, reg)
            losses.append(L)

            # Perform parameter update
            self.W += -eta*G
          

            if verbose and it % verb_steps == 0:
                print 'iteration %d: loss %f' % (it, loss)

        return losses

    def predict(self, X):
        """ Uses the trained weights of the linear classifier to predict labels
            for data points.

            @param X    D x N array of training data
            @return     Predicted labels for the data in X
        """
        y = np.zeros(X.shape[1])
        y = np.argmax(np.dot(self.W, X), axis=0)
        return y
  
    def loss(self, X_batch, y_batch, reg):
        """ Computes the loss function and its derivative. Subclasses will
            override this.

            @param X_batch  D x N array of data
            @param y_batch: N length array of training labels with values 0, 1,
                            ..., K-1 for K classes
            @param reg:     Regularization strength
            @returns:       A tuple containing:
                            - loss as a single float
                            - gradient with respect to self.W
        """
        pass


class SVM(LinearClassifier):
    """ Encapsulates a multiclass linear SVM.
    """
    def loss(self, X_batch, y_batch, reg):
        return self.svm_loss(X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ Encapsulates a multiclass linear softmax + cross-entropy classifier.
    """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss(self.W, X_batch, y_batch, reg)


def svm_loss(W, X_batch, y_batch, reg):
    pass


def softmax_loss(W, X_batch, y_batch, reg):
    pass


