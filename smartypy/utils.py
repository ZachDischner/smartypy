#!/usr/bin/env python

__author__     = 'Zach Dischner'
__copyright__  = ""
__credits__    = ["NA"]
__license__    = "NA"
__version__    = "0.0.2"
__maintainer__ = "Zach Dischner"
__email__      = "zach.dischner@gmail.com"
__status__     = "Dev"
__doc__        ="""
File name: utils.py
Created:  04/Sept/2016
Modified: 30/Sept/2016

Houses utilities and common functions used across various scripts

"""

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from numba import jit, njit


###### Module variables
_here    = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(_here)[0])   # Absolute version of sys.path.append(..). Helpful so we can run this module standalone...



##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
## @njit on these speed it up from 91us to 3us. Cool! About 9x faster than matlab
@njit
def _normalize_feature(x):
    """Normalize a feature to zero mean, 1 std range

    Algorithm:
                       (x[i] - mu)
        x_norm[i]  =  -------------
                         sigma

    Args:
        x:  Feature vector to normalize

    Returns:
        x_norm: Normalized feature vector
        mu:     Computed mean of normalization
        sigma:  Computed standard deviation of feature

    >>> _normalize_feature(np.array([2104, 1600, 2400, 1416]))
    (array([ 0.57160715, -0.71450894,  1.32694517, -1.18404339]), 1880.0, 391.8775318897474)
    """
    sigma  = x.std() # Kinda confused about dimensional normalization. ddof=1 matches matlab's default. Can't do with numba
    mu     = x.mean()
    if sigma == 0:
        x_norm = x*1.0 # Do this so we can jit, makes sure x_norm is always float
    else:
        x_norm = (x-mu)/sigma
    return x_norm, mu, sigma

@njit
def normalize_features(X):
    """Normalize a feature array. See _normalize_feature
    """
    n = X.shape[1]
    Xn = np.zeros(X.shape)
    mu, sigma = np.zeros(n),np.zeros(n)

    ## Smarter way to map() this or something??
    # Figure this is pretty fast and readable. Could do same thing with a comprehension but
    # reconstruction is pretty ugly
    for idx in range(n):
        Xn[:,idx],mu[idx], sigma[idx] = _normalize_feature(X[:,idx])
    return Xn, mu, sigma

def is_normalized(X):
    if False in list(map(lambda x: np.isclose(x,0), np.mean(X,axis=0))):
        return False
    if False in list(map(lambda x: np.isclose(x,1), np.std(X,axis=0))):
        return False
    return True
    
def denormalize(Xn, mu, sigma):
    """Denormalize features, get back to starting point

    Same logic works for single vector feature (x) and matrix of feature columns (X)
    """
    X = Xn*sigma + mu
    return X

def polymap(X, degree=1):
    """Using pro library:
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

    Adds offest column (1) if it doesn't exist. 

    Note that if X does not have the (x0) offset column, polymap(X,degree=1) will return X with that
    offset column added. if X does have the (x0) offset column, polymap(X,degree=1) will return the 
    exact same matrix.

    If degree is None, X is returned as is, no matter what. 

    Given Design Matrix X = [[x0],[x1],[x2]], This creates new design matrix with all polynomial terms
    up to the `order`th power:
        [[x0], [x1], [x2], [x1^2], [x1*x2], [x2^2], [x1^3] ... [x1*x2^(order-1)],[x2^(order)]]

    Examples:
    >>> polymap([[1,4],[2,5],[3,6]],degree=2)
    array([[  1.,   1.,   4.,   1.,   4.,  16.],
           [  1.,   2.,   5.,   4.,  10.,  25.],
           [  1.,   3.,   6.,   9.,  18.,  36.]])
    """
    if degree is None:
        return X

    if not isinstance(X,np.ndarray):
        X = np.array(X)
 
    non_offset_start = 1 # This assumes that there is already an offset column of ones. Only polymap X[:,1:]
    ## If working with 1d array, don't bother with offset column? Used in plot_decision_boundary only)
    if X.ndim <= 1:
        # PolynomialFeatures needs 2d array to work properly
        # utils.printYellow("polymap should take a 2d array, even for single vectors: np.array([[1,2,3]]). Circumventing for you anyways")
        X = np.array([X])

    if False in (X[:,0]==1):
        non_offset_start = 0 # Turns out, no offset colum exists. So map the whole thing

    poly = PolynomialFeatures(degree=degree,include_bias=True)
    Xp   = poly.fit_transform(X[:,non_offset_start:])

    return X

@njit
def sigmoid(z):
    """ Compute sigmoid function on z. 

    Examples:
    >>> sigmoid(0)
    0.5
    """
    return 1/(1+np.exp(-z))

@njit
def sigmoid_prime(z):
    """ Sigmoid function Derivative
    Examples:
    >>> sigmoid_prime(0)
    0.25
    """
    return sigmoid(z) * (1-sigmoid(z))

@njit
def hypothesize(X,theta):
    """Hypothesize predicion y given dataset X and coefficient vector theta

    Args:
        X:      (ndarray Reals) Design Matrix 
        theta:  (vector Reals) Coefficient Vector

    Returns:
        hypothesies:    (Real) y prediction
    """
    return sigmoid(X @ theta) 

def _dimensionalize_y(y):
    """Make sure y is a single dimension vector

    This is tricksy and a little dumb IMO. If y is (5000x1), it takes 500x longer to 
    do the calculations below. So convert it to a (5000,) vector...
    """
    if y.ndim > 1:
        print("Cost calculation with multi dimensional is dramatically slower than with a 1d vector! You should use y=y[:,0] for speed...")
        max_idx = y.shape.index(max(y.shape))
        if max_idx == 0:
            y = y[:,0]
        else:
           y = y[0,:] 

    return y

## Colors!!!
class bcolors:
    HEADER  = '\033[95m'
    OKBLUE  = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'

def printColor(msg,color):
    print(color + str(msg) + bcolors.ENDC)

def printYellow(msg):
    printColor(msg,bcolors.WARNING)
def printGreen(msg):
    printColor(msg,bcolors.OKGREEN)
def printBlue(msg):
    printColor(msg, bcolors.OKBLUE)
def printRed(msg):
    printColor(msg,bcolors.FAIL)























