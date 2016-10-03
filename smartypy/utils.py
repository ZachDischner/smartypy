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























