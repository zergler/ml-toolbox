#!/usr/bin/env python2

""" Implements a linear classifier base class which can be inherited by linear
    models.
"""

import random

import matplotlib.pyplot as plt
import numpy as np


class LinearClassifier:
    """ Encapsulates a linear classifier.
    """

    def __init__(self):
        """ Initializes the linear classifier.
            
        """
        self.W = None

    def init_weights_std_normal(self, K, D, sigma):
        """ Initializes the weights using a standard normal .
        """
        # Randomly initialize weights
        self.W = np.random.randn(K, D)*sigma

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
            self.init_weights_std_normal(K, D, 0.001)

        # Give a reasonable step size for displaying epochs
        verb_steps = epochs/10

        losses = []
        pop = range(N)
        for i in xrange(epochs):
            # Grab some batches
            indices = random.sample(pop, s)
            X_batch = X[:, indices]
            y_batch = y[indices]

            # Evaluate loss and gradient
            (L, G) = self.loss(X_batch, y_batch, reg)
            losses.append(L)

            # Perform parameter update
            self.W += -eta*G

            if verbose and i % verb_steps == 0:
                print 'iteration %d: loss %f' % (i, L)

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

X_train = None
X_val = None
X_test = None
y_train = None
y_val = None
y_test = None

def _test_linear_classifier():
    # Global variables
    global X_train
    global X_val
    global X_test
    global y_train
    global y_val
    global y_test

    # Load data to use
    filenames = ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
                 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte']
    filenames = ['../datasets/MNIST/' + i for i in filenames]
    (train, test) = dataset.load_mnist_all(filenames[0], filenames[1], filenames[2], filenames[3])

    X = np.array(train[0]).T
    y = np.array(train[1])

    # Form a training and validation set from the training data
    ((X_train, y_train), (X_val, y_val)) = preprocess.split_train_val(X, y)

    X_test = np.array(test[0]).T
    y_test = np.array(test[1])

    # Visualize the mean image
    mean_image = np.mean(X_train, axis=1)
    plt.figure()
    plt.imshow(mean_image.reshape((28, 28)).astype('uint8'), interpolation='none', cmap='gray')
    plt.axis('off')
    plt.show()

    # Subtract the mean from the image data
    X_train = (X_train.T - mean_image).T
    X_val = (X_val.T - mean_image).T
    X_test = (X_test.T - mean_image).T

    # Append the bias dimension to the data.
    X_train = np.vstack([X_train, np.ones((1, X_train.shape[1]))])
    X_val = np.vstack([X_val, np.ones((1, X_val.shape[1]))])
    X_test = np.vstack([X_test, np.ones((1, X_test.shape[1]))])

    # Test the svm
    _test_svm()


def _test_svm():
    import svm

    # Global variables
    global X_train
    global X_val
    global X_test
    global y_train
    global y_val
    global y_test

    # Train the SVM
    svm = svm.LinSVM()
    tic = time.time()
    losses = svm.train_sgd(X_train, y_train, eta=1e-7, reg=5e4, epochs=1500, s=250, verbose=True)
    toc = time.time()
    print 'that took %fs' % (toc - tic)

    # Plot the training error
    plt.plot(losses)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    # Get the training and validation accuracy
    y_train_pred = svm.predict(X_train)
    print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
    y_val_pred = svm.predict(X_val)
    print 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), )

    # Visualize the weights 
    classes = [str(i) for i in range(svm.W.shape[0])]
    misc.viz_weights_categorical(svm.W, classes, (28, 28))


def main():
    _test_linear_classifier()


if __name__ == '__main__':
    import pdb
    import time
    import dataset
    import preprocess
    import misc
    main()
