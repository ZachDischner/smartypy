#!/usr/bin/env python

__author__     = 'Zach Dischner'
__copyright__  = ""
__credits__    = ["NA"]
__license__    = "NA"
__version__    = "0.0.0"
__maintainer__ = "Zach Dischner"
__email__      = "zach.dischner@gmail.com"
__status__     = "Dev"
__doc__        ="""
File name: neuralNetwork.py
Created:  15/Sept/2016
Modified: 15/Sept/2016

Notes:
    For all Neural Network utitlities, it is assumed that the offset column in
    design matrix and hidden activation layers has *not* been added. They will
    add the column for you. No checks are performed.

Nomenclature:
    See README.md ==> Notation Convention section for an explanation of variables.
Examples:


TODO:
    """
##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import numpy as np
import pandas as pd
import pylab as plt
import scipy.io as sio
import pytest
from scipy.optimize import minimize, fmin_cg

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
from smartypy import utils, _SMARTY_DIR
from smartypy.logisticRegression import sigmoid, hypothesize # consolodate these to some other function?

###### Module variables

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def activate(a,theta):
    """Process activation matrix `a`(j-1) through next layer with weights vector characterized
    by theta.

    a(j+1) = sigmoid( theta @ a(j-1))

    Args:
        theta:  (Array Reals) Weights Matrix (Includes weight of bias node)
        a:      (Array Reals) Activation Matrix

    Returns:
        a_next: (Array Reals) Activation layer of next (j+1)th activation layer (does not include bias term)

    Examples:
        (Logical AND)
        >>> round( activate( np.array([0,1]), np.array([-30,20,20])))
        0

        >>> round( activate( np.array([1,1]), np.array([-30,20,20])))
        1
    """
    ## If just 1d vector, re-wrap
    axis = 1
    if a.ndim == 1:
        axis = 0
    a_offset = np.insert(a,0,1,axis=axis)
    z = a_offset @ theta.T
    a_next = sigmoid(z)
    return a_next

def _test_prediction():
    """Comprehensive unit test would be nice. For now, just perform the same test procedure
    as MATLAB neuralNetworkPredictiontest.m
    """
    print("\n\n==============================Begin Neural Network Prediction Test============================\n")

    ###### Load X,y and weights from Matlab saved mat files
    dataset = os.path.join(_SMARTY_DIR,"test","data","ex3data1.mat")
    weights = os.path.join(_SMARTY_DIR,"test","data","ex3weights.mat")
    mat = sio.loadmat(dataset)
    wmat = sio.loadmat(weights)

    X = mat['X']
    # Load y specifically into 1d array with datatype int16 (from mat file, it is an uint8. Which can cause tricksy issues)
    y = mat['y'][:,0].astype('int16')
    y[y==10] = 0
    ## Classifiers are shifted by 1 to the array index, since the theta training was done in Matlab with 1 indexing
    classifiers = [1,2,3,4,5,6,7,8,9,0]
    theta1 = wmat["Theta1"] # provided already trained weights
    theta2 = wmat["Theta2"]

    ###### Form a Neural Network
    a0 = activate(X,theta1)
    a1 = activate(a0,theta2)
    # Thats it!
    hypothesis = a1
    ## Hypothesis has in each row, the probabilities that the sample (corresponding row in X) is
    # of the classification in the corresponding classification list. For us, the classifications
    # are easy, and correspond to the indices of that hypothesis vector.
    # I.E. for h=hypothesis[n], max(h) == h[3] means that classification 3 (which is also 3) is most probable
    pred_idx = hypothesis.argmax(axis=1)
    predictions = [classifiers[idx] for idx in pred_idx]
    accuracy = (predictions==y).mean() * 100.0
    print("Neural Network Training Accuracy: {}\t\t[MATLAB: {}]".format(accuracy,97.52))
    print("\n\n==============================End Neural Network Prediction Test============================\n")
    return X,y,theta1,theta2,hypothesis,predictions


##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    test()

    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')
