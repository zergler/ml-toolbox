#!/usr/bin/env python2


""" Contains useful miscellaneous functions.
"""

import itertools
import numpy as np


def viz_weights_categorical(W, classes, shape):
    """ Displays the weights of the weight matrix as images.

        @param W        Weight matrix of size (num_classes x num_dims)
        @param classes  Names of the classes
        @param shape    Shape of each weight vector in image form (tuple)
    """
    # Make sure the shape is correct
    assert(reduce(lambda x, y: x*y, shape) == W.shape[1])

    # Depending on the choice of learning rate and regularization strength,
    # these may or may not be nice to look at.
    w = svm.W[:, :-1]  # strip out the bias
    w = np.reshape(w, tuple([W.shape[0]] + list(shape)))  # reshape to image form
    (w_min, w_max) = np.min(w), np.max(w)
    for i in xrange(W.shape[0]):
        # Determine the best way to display the images (favoring cols)
        shape_opt = get_opt_viz_shape(W.shape[0])
        plt.subplot(min(shape_opt), max(shape_opt), i + 1)
          
        # Rescale the weights to be between 0 and 255
        wimg = 255.0*(w[i].squeeze() - w_min)/(w_max - w_min)
        plt.imshow(wimg.astype('uint8'), interpolation='none')
        plt.axis('off')
        plt.title(classes[i])


def get_opt_viz_shape(N):
    """ Computes the optimal vizualization structure for a bunch of images.
        
        Example: Say there are 10 classes that you want to vizualize the weights
        for. You could just display them in a row, but this would be pretty
        ugly. Instead this algorithm computes a matrix containing the images
        which has a shape that maximizes the 'squareness' and minimizes the
        number of missing cells.

        @param N    Number of images to vizualize
        @return     A 2-tuple containing the optimal shape
    """
    AN = [N, N+1]
    AP = []
    PP = []
    AP.append(get_opt_pair(AN[0]))
    AP.append(get_opt_pair(AN[1]))
    #PP.append(int(is_prime(N)))
    #PP.append(int(is_prime(N + 1)))
    D = [AP[0][1], AP[1][1]]

    i = 1

    # Need to replace stopping condition with a better one but works as long as
    # number of iterations is high enough to find the optimal solution
    while (i < 10) or is_prime(AN[i]):
        i = i + 1
        AN.append(AN[i - 1] + 1)
        #PP.append(int(is_prime(AN[i])))
        AP.append(get_opt_pair(AN[i]))
        D.append(AP[i][1] + i)
    
    D = np.array(D)
    shape_opt = AP[D.argmin()][0]
    #blah = np.zeros(100)
    #blah[D.argmin()] = 1
    #print 'Opt shape: ', shape_opt, "\nN's: ", AN, "\nLocation: ", blah, "\nPrime: ", PP, "\nP's: ", AP, "\nD's: ", D
    return shape_opt

def get_opt_pair(N):
    """ Computes all pairs p_k = (f_i, f_j) such that f_i, f_j are factors of
        N and f_i*f_j = N. Then finds the optimal pair p* with optimal
        difference d* = min(diff(p_k)) for all valid pairs p_k where diff(p_k)
        = abs(f_i - f_j).

        @param N    Number for which optimal pair and distance are being
                    computed for
        @return     2-tuple containing optimal pair p* and optimal distance d*
    """
    F = get_factors(N)
    P = get_valid_pairs(F, N)
    D = np.array([abs(f1 - f2) for (f1, f2) in P])
    d_opt = D.min()
    p_opt = P[D.argmin()]
    return (p_opt, d_opt)

def get_valid_pairs(F, N):
    """ Computes the valid pairs P of N given a list containing all (non unique)
        factors F of N.
    """
    P = set([(f1, f2) for (f1, f2) in list(set(itertools.combinations(F, 2))) if f1*f2 == N])
    return list(P)


def get_factors(N):
    """ Computes all possible (non-unique) factors of N.

        @param  N   Number for which to compute factors for
        @return     All possible (non-unique factors
    """
    return reduce(list.__add__, ([i, N//i] for i in range(1, int(N**0.5) + 1) if N % i == 0)) 


def is_prime(n):
    if n < 2: return False
    return all(n%i for i in itertools.islice(itertools.count(2), int(np.sqrt(n)-1)))


def _test_get_factors():
    from random import randint
    sample_size = 25
    N = [randint(1, 100) for i in range(sample_size)]
    F = [get_factors(i) for i in N] 
    for (i, f) in zip(N, F):
        print 'Integer: ', i, 'Factors: ', f


def _test_get_valid_pairs():
    from random import randint
    sample_size = 25
    AN = [randint(1, 100) for i in range(sample_size)]
    AF = [get_factors(N) for N in AN] 
    AP = [get_valid_pairs(F, N) for (F, N) in zip(AF, AN)] 
    for (N, F, P) in zip(AN, AF, AP):
        print 'Integer: ', N, '\t Factors: ', F, '\t Valid pairs: ', P


def _test_get_opt_pair():
    from random import randint
    sample_size = 25
    AN = [randint(1, 100) for i in range(sample_size)]
    AF = [get_factors(N) for N in AN] 
    AP = [get_valid_pairs(F, N) for (F, N) in zip(AF, AN)] 
    AO = [get_opt_pair(N) for N in AN]
    for (N, F, P, O) in zip(AN, AF, AP, AO):
        print 'Integer: ', N, '\t Factors: ', F, '\t Valid pairs: ', P, '\t Opt pair: ', O


def _test_get_opt_viz_shape():
    N = 531
    shape = get_opt_viz_shape(N)


def main():
   import pdb 
   _test_get_opt_viz_shape()


if __name__ == '__main__':
    main()
