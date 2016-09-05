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
        m:      Number of examples/samples. (Includes x0 feature)
        x:      Feature vector dataset (m x 1)
        X:      Feature Matrix (m x n)
        Xn:     Normalized Feature Matrix (m x n)
        y:      Target/Solution vector (m x 1)
        J:      Cost of a sample (single value)
        theta:  Linear Regression Coefficient Vector (n x 1)
        h:      Hypothesis of form: h(theta) = X @ theta

TODO:
    * Type/vector size error handling?
    * Optimizations, @njit,
    * Refactor for infix `@` operator
    * Unit tests!! *doctest for small functions? pytest for bigger ones
    * Add J(theta) contour plot

    """

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import numpy as np
import pandas as pd
import pylab as plt
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
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta

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

def gradient_descent(Xn,y,theta,alpha,num_iters=1000,tol=None):
    """Perform gradient descent optimization to learn theta that creates the best fit
    hypothesis h(theta)=X @ theta to the dataset

    Args:
        Xn:     Normalized Feature Matrix
        y:      Target Vector
        alpha:  (Real, >0) Learning Rate

    Kwargs:
        num_iters:  (Real) Maximum iterations to perform optimization
        tol:        (Real) If provided, superscede num_iters, breaking optimization if tolerance cost is reached
    """
    
    ## Check to see if Xn is normalized. Warn if not. 
    if round(Xn[:,1].std()) != 1:
        utils.printRed("Gradient Descent X matrix is not normalized. Pass in normalized in the future to ensure convergence")
        Xn,_,_ = normalize_features(Xn)

    m = 1.0*len(y)
    J_history =[]
    for idx in range(0,num_iters):
        ## Compute new theta
        theta = theta -  (alpha/m) * ((Xn @ theta - y).T @ Xn).T

        ## Save new J cost
        J_history.append(compute_cost(Xn,y,theta))
        if (tol is not None) and (J_history[-1] <= tol):
            break

        ## Check to make sure J is decreasing...
        if (idx > 1) and J_history[-2] <= J_history[-1]:
            utils.printRed("Gradient Descent is not decreasing! Alpha: {}\t previous J {}\tJ {}. Try decreasing alpha".format(alpha,J_history[-2], J_history[-1]))

    return theta, J_history

def plot_3d(X,y,theta=None,theta_norm=None, xlabel="x",ylabel="y", zlabel="z"):
    """Vis helper for 3d datasets

    only helpful for n=2 feature learning problems

    No catches for wrong dimensions, mismatched dataset, or too big of Xs for Normal Equation...

    Args:
        X:  [Matrix] Unnormalized Feature Matrix X
    """
    m = len(y)
    if theta is None:
        theta, J_History = gradient_descent(X,y,[0,0,0],0.01)

    ###### First, plot hypothsis
    xs = np.linspace(X[:,1].min(), X[:,1].max(), m)
    ys = np.linspace(X[:,2].min(), X[:,2].max(), m)
    Xh = np.ones(X.shape); Xh[:,1] = xs; Xh[:,2] = ys
    Xhn, mu, sigma = normalize_features(Xh)
    zs = Xhn @ theta

    if theta_norm is None:
        theta_norm = solve_normal(X,y)
        normal_zs = Xh @ theta_norm

    ###### Plot it! 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,1],X[:,2],zs=y, label="Sample Data")
    ax.plot(xs,ys,zs=zs, label="Hypothesis")
    ax.plot(xs,ys,zs=normal_zs, label="Normal Hypothesis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.show(block=False)
    return ax


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

