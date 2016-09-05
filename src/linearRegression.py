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
    Currently supports Python3.5 only. Main reason is to use the new 3.5 `@` infix
    matrix math operator. Otherwise `X.dot(theta)` gets pretty cumbersome.
    Setup a new bare bones environment using conda or use teh accompanying environment.yml
    to set up a compatible environment using:
        $ conda env create -f environment.yml
        $ source activate python35

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
        h:      Hypothesis of form: h(theta) = X @ theta

TODO:
    * Type/vector size error handling?
    * Optimizations, @njit,
    * Refactor for infix `@` operator
    * Unit tests!! *doctest for small functions? pytest for bigger ones

    """

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import numpy as np
import pandas as pd
from numba import jit, njit

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _here)
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
    X_norm = np.zeros(X.shape)
    mu, sigma = np.zeros(n),np.zeros(n)

    ## Smarter way to map() this or something??
    # Figure this is pretty fast and readable. Could do same thing with a comprehension but
    # reconstruction is pretty ugly
    for idx in range(n):
        X_norm[:,idx],mu[idx], sigma[idx] = _normalize_feature(X[:,idx])
    return X_norm, mu, sigma

def denormalize(X_norm, mu, sigma):
    """Denormalize features, get back to starting point

    Same logic works for single vector feature (x) and matrix of feature columns (X)
    """
    X = X_norm*sigma + mu
    return X

def gradient_descent(X,y,theta,alpha,num_iters=1000,tol=None):
    """Perform gradient descent optimization to learn theta that creates the best fit
    hypothesis h(theta)=X @ theta to the dataset

    Args:
        X:
        y:
        alpha:  Learning Rate

    Kwargs:
        num_iters:  (Real) Maximum iterations to perform optimization
        tol:        (Real) If provided, superscede num_iters, breaking optimization if tolerance cost is reached
    """
    m = 1.0*len(y)
    J_history =[]
    for iter in range(0,num_iters):
        ## Compute new theta
        theta = theta -  (alpha/m) * ((X @ theta - y).T @ X).T
        #print("theta: {}".format(theta))

        ## Save new J cost
        J_history.append(compute_cost(X,y,theta))
        if (tol is not None) and (J_history[-1] <= tol):
            break
    return theta, J_history

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
    ## Load data into dataframe
    df = pd.read_csv("../test/data/ex1data2.txt",names=["area","rooms","price"])
    X = np.array(df.iloc[:,0:2])
    y = np.array(df.price)

    ## Prepend the theta0 column to X
    X = np.insert(X, 0, 1, axis=1)
    
    theta = np.zeros(X.shape[1])
    return X, y, theta

