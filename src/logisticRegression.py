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
File name: logisticRegression.py
Created:  11/Sept/2016
Modified: 11/Sept/2016

Description:
    Multivariate Logistic Regression utilities written to mimic Matlab/Octave
    scripts developed for the Coursera Machine Learning course.
    Currently supports Python3.5 only. Main reason is to use the new 3.5 `@` infix
    matrix math operator. Otherwise `X.dot(theta)` gets pretty cumbersome.
    Setup a new bare bones environment using conda or use the accompanying environment.yml
    to set up a compatible environment using:
        $ conda env create -f environment.yml
        $ source activate python35

Note:
    Error checking, vector sizes, etc are omitted for the time being.

Nomenclature:
    Variables and nomenclature follows the same convention as specified in
    Machine Learning course work. Outlined here so as to avoid repition in
    function definitions

        n:      Number of features (excluding x0 feature)
        m:      Number of examples/samples
        x:      Feature column vector dataset (m x 1)
        X:      Feature or Design Matrix (m x n+1)
        Xn:     Normalized Feature Matrix (m x n+1)
        y:      Target/Solution vector (m x 1)
        J:      Cost of a sample (single value)
        theta:  Linear Regression Coefficient Vector (n+1 x 1) ==> theta0 + theta1*x1 + theta2*x2 ... + thetan*xn
        h:      Hypothesis of form: h(X) = X @ theta
                                    h(x) = theta.T @ x ==> [ --- theta --- ] @ [x]

TODO:
    * Type/vector size error handling?
    * Optimizations, @njit,
    * Unit tests!! *doctest for small functions? pytest for bigger ones

    """

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import numpy as np
import pandas as pd
import pylab as plt
from mpl_toolkits.mplot3d import axes3d
from numba import jit, njit
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
_smarty_dir =  os.path.split(_here)     # Always .. from smarty files
sys.path.insert(0, _here)
import utils
from linearRegression import normalize_features

np.set_printoptions(suppress=False)
np.set_printoptions(precision=5)

###### Module variables

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def sigmoid(z):
    """ Compute sigmoid function on z. 

    Examples:
    >>> sigmoid(0)
    0.5
    """
    return 1/(1+np.exp(-z))

def hypothesize(X,theta):
    """Hypothesize predicion y given dataset X and coefficient vector theta

    Args:
        X:      (ndarray Reals) Design Matrix 
        theta:  (vector Reals) Coefficient Vector

    Returns:
        hypothesies:    (Real) y prediction
    """
    return sigmoid(X @ theta) 

def compute_cost(X,y,theta,lam=1.0):
    """Compute cost of hypothesized `theta` against test dataset X and solutions y

    Description:
        Cost is calculated as:
            -log(h(x))      if y == 1
            -log(1-h(x))    if y == 0

        Regularization of cost function occurs for any lambda > 0

    Args:
        X:      (ndarray Reals) Design Matrix 
        y:      (vector Reals) Solution vector
        theta:  (vector Reals) Coefficient Vector
    
    Kwargs:
        lam:    (Real >= 0) Lambda "regularization parameter"

    Returns:
        J:      (Real) Cost of hypothesis
        grad:   ([m x 1] vector Reals) Gradient of J wrt thet. [d(J)/d(theta_j)]
    """
    m = len(y)
    n = len(theta)
    hypothesis = hypothesize(X,theta)
    
    ## Cost function
    #                                                                       |retularization      | This part subustitute the theta0 term with zero in the theta array since it does not get regularized
    J = (1.0/m) * sum(-y*np.log(hypothesis) - (1-y)*(np.log(1-hypothesis))) + (lam/2.0/m)*sum(np.insert(theta[1:],0,0)**2);

    return J

def compute_gradient(X,y,theta,lam=1.0):
    """Compute Regularized gradient of cost function. See compute_cost for argument details"""
    ## Gradient of cost function
    m = len(y)
    n = len(theta)
    hypothesis = hypothesize(X,theta)
    grad = []
    for jj in range(n):
        grad.append( (1.0/m) * sum((hypothesis - y)*X[:,jj]))

    ## Regularization bit. 
    #                            | This part subustitute the theta0 term with zero in the theta array since it does not get regularized
    grad += (lam/m) * np.insert(theta[1:],0,0)

    return grad


def polymap(X, degree=6):
    """Using pro library:
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

    Adds offest column (1) if it doesn't exist. Only works for adding polynomial features to design matrices X
    with 2 independant features.

    Given Design Matrix X = [[x0],[x1],[x2]], This creates new design matrix with all polynomial terms
    up to the `order`th power:
        [[x0], [x1], [x2], [x1^2], [x1*x2], [x2^2], [x1^3] ... [x1*x2^(order-1)],[x2^(order)]]

    Examples:
    >>> polymap([[1,4],[2,5],[3,6]],degree=2)
    array([[  1.,   1.,   4.,   1.,   4.,  16.],
           [  1.,   2.,   5.,   4.,  10.,  25.],
           [  1.,   3.,   6.,   9.,  18.,  36.]])
    """
    if not isinstance(X,np.ndarray):
        X = np.array(X)

    add_offset_col = False
    if False in (X[:,0]==1):
        add_offset_col = True

    poly = PolynomialFeatures(degree=degree,include_bias=add_offset_col)
    Xp   = poly.fit_transform(X)

    return Xp

def solve(X,y,theta=None,poly_degree=6,lam=1):  
    """Solve a logistic regression problem represented by Design Matrix X and results y
    
    Prints a big red warning if minimization did not occur. Tries a few differnet methods. 

    Could always write our own, or use a recommended solver from Python Coursera https://github.com/royshoo/mlsn/blob/master/python/courseraEx02.py

    Returns:
        cost:   (real) Cost at minimized value of theta
        theta:  (array reals) theta vector which minimizes cost function
    """
    if poly_degree is not None:
        X = polymap(X, degree=poly_degree)

    if theta is None:
        theta = np.zeros(X.shape[1])

    print("Solving for minimum-cost theta using initial conditions theta = {:.15}...\nAnd order {} polynomial mapped X matrix".format(str(theta),poly_degree))
    
    ## Simple wrapper function to make the call to get J,cost a function with just one input.
    # I think it is cleaner than mucking up the call to minimization functions with Kwargs and whatnot
    def func(theta):
        return compute_cost(X,y,theta,lam=lam),compute_gradient(X,y,theta,lam=lam)

    
    ## The 'Truncated Newton' TNC method seems to work best for minmimization
    res = minimize(func, theta, method='TNC', jac=True)

    if res.success is True:
        utils.printGreen("Cost minimization using 'TNC' method has succeeded with minimum cost {:.5} for theta = {:.35}...".format(res.fun,str(res.x)))
        return res.fun, res.x
    else:
        utils.printYellow("Minimization using 'TNC' failed! Trying 'BFGS' method")
        res = minimize(func, theta, method='BFGS', jac=True)
        if res.success is True:
            utils.printGreen("Cost minimization using 'BFGS' method has succeeded with minimum cost {} for theta = {:.25}...".format(res.fun,(res.x)))
        else:
            utils.printRed("Logistic Regression cost minimization failed using two methods. Maybe try some others?")

    return res.fun, res.x
    



def gradient_descent(Xn,y,theta,alpha,num_iters=1000,tol=None,theta_hist=False):
    """Perform gradient descent optimization to learn theta that creates the best fit
    hypothesis h(theta)=X @ theta to the dataset

    Args:
        Xn:     Normalized Feature Matrix
        y:      Target Vector
        alpha:  (Real, >0) Learning Rate

    Kwargs:
        num_iters:  (Real) Maximum iterations to perform optimization
        tol:        (Real) If provided, superscede num_iters, breaking optimization if tolerance cost is reached
        theta_hist: (Bool) IF provided, also return theta's history
    """
    
    # Check to see if Xn is normalized. Warn if not. 
    if round(Xn[:,1].std()) != 1:
        utils.printYellow("Gradient Descent X matrix is not normalized. Pass in normalized in the future to ensure convergence")
        # Xn,_,_ = normalize_features(Xn)

    m = 1.0*len(y)
    J_history =[]
    theta_history = []
    for idx in range(0,num_iters):
        ## Compute new theta
        theta = theta -  (alpha/m) * ((Xn @ theta - y).T @ Xn).T
        theta_history.append(theta)

        ## Save new J cost
        J_history.append(compute_cost(Xn,y,theta))
        if (idx>1) and (tol is not None) and (J_history[-1]-J_history[-2] <= tol):
            break

        ## Check to make sure J is decreasing...
        if (idx > 1) and J_history[-2] <= J_history[-1]:
            utils.printRed("Gradient Descent is not decreasing! Alpha: {}\t previous J {}\tJ {}. Try decreasing alpha".format(alpha,J_history[-2], J_history[-1]))
    if theta_hist:
        return theta, J_history, np.vstack(theta_history)
    return theta, J_history


def test():
    """Comprehensive unit test would be nice. For now, just perform the same 

    """
    ## Maybe this! Save everything from MATLAB verified into mat file, read and compare
    # var = sio.loadmat("test/polyTrue.mat")
    # X = var['X'] >> loads into np array X! Neat!

    ## Load data into dataframe
    df = pd.read_csv("../test/data/ex2data1.txt",names=["test1","test2","passfail"])
    X = np.array(df.iloc[:,0:2])
    y = np.array(df["passfail"])

    ## Prepend the theta0 column to X
    X = np.insert(X, 0, 1, axis=1)

    theta = np.zeros(X.shape[1])

    ####### Solutions come from MATLAB
    J_true = 0.693147180559946
    return X, y, theta

