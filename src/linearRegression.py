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
File name: linearRegression.py
Created:  04/Sept/2016
Modified: 04/Sept/2016

Description:
    Multivariate Linear Regression utilities written to mimic Matlab/Octave
    scripts developed for the Coursera Machine Learning course.
    Currently supports Python3 only. Use accompanying environment.yml to
    set up a compatible environment using
        $ conda ....

Note:
    Error checking, vector sizes, etc are omitted for the time being.

Nomenclature:
    Variables and nomenclature follows the same convention as specified in
    Machine Learning course work. Outlined here so as to avoid repition in
    function definitions

        n:      Number of features
        m:      Number of examples/samples
        x:      Feature dataset (m x 1)
        y:      Sample solution (m x 1)
        X:      Matrix of Feature columns (m x n)
        J:      Cost of a sample (single value)
        theta:  Linear Regression Coefficient Vector (n x 1)

TODO:
    * Type/vector size error handling
    * Optimization

    """

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import numpy as np
from numba import jit, njit

## Local utility module
import utils
###### Module variables

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def compute_cost(X,y,theta):
    """Compute cost of hypothesized `theta` against test dataset X and solutions y

    Description:
        Cost is calculated as the sum of the square of the difference between
        hypothesis and dataset divided by twice the number of examples.

    Args:
        X:      <official name?>(ndarray Reals)
        y:      <official name?>(vector Reals)
        theta:  <official name?>(vector Reals)

    Returns:
        J:  (Real) Cost of hypothesis
    """
    m = len(y)
    hypothesis = X.dot(theta)
    error = (hypothesis - y)**2.0
    J = (1.0/2.0/m) * sum(error)
    return J

def solve_normal(X,y):
    """Solve Normal equations for analytical, closed form minimization
    of cost function J

    Algorithm:
        Standard least squares minimization right?
        inv(X'*X) * X * y
    Args:
    --

    Returns:
        theta:
    """
    ## Could one line it, but man it is ugly with numpy matrix syntax...
    Xt = X.transpose()
    _component = np.linalg.pinv(Xt.dot(X))
    return _component.dot(Xt).dot(y)

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
    """
    sigma  = x.std() # Kinda confused about dimensional normalization. ddof=1 matches matlab's default
    mu     = x.mean()
    x_norm = (x-mu)/(sigma or 1)
    return x_norm, mu, sigma

@njit
def normalize_features(X):
    """Normalize a feature array. See _normalize_feature"""
    n = X.shape[1]
    X_norm = np.zeros(X.shape)
    mu, sigma = np.zeros(n),np.zeros(n)

    ## Smarter way to map() this or something??
    for idx in range(n):
        X_norm[:,idx],mu[idx], sigma[idx] = _normalize_feature(X[:,idx])
    return X_norm, mu, sigma

def denormalize_features(X_norm, mu, sigma):
    """Denormalize features, get backto starting point"""
    X = X_norm*sigma + mu

def test():
    """Comprehensive Unit Tests? How about a pickle file with X, y, theta
    and comparative solutions for the same dataset given by Matlab?

    For now...
    X=np.array([[1,2104,3],[1,1600,3],[1,2400,3],[1,1416,2]])
    y=np.array([399900, 329900, 369000, 232000])
    theta = 1e4*np.array([8.959790954478693, 0.013921067401755, -0.873801911287263])

    true cost: 7.522274051241411e+08
    normal:    1.0e+04 * [-4.698317251656216, 0.005848790585483, 9.808214891250811]
    """
    pass
