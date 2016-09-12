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
from numba import jit, njit
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
_smarty_dir =  os.path.split(_here)[0]     # Always .. from smarty files
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
    ## If working with 1d array, don't bother with offset column? Used in plot_decision_boundary only)
    if X.ndim > 1:
        if False in (X[:,0]==1):
            add_offset_col = True
    else:
        utils.printYellow("polymap should take a 2d array, even for single vectors: np.array([[1,2,3]]). Circumventing for you anyways")
        return polymap(np.array([X]))

    poly = PolynomialFeatures(degree=degree,include_bias=add_offset_col)
    Xp   = poly.fit_transform(X)

    return Xp

def solve_regression(X,y,theta=None,poly_degree=6,lam=1):  
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
            utils.printGreen("Cost minimization using 'BFGS' method has succeeded with minimum cost {} for theta = {:.25}...".format(res.fun,str(res.x)))
        else:
            utils.printRed("Logistic Regression cost minimization failed using two methods. Maybe try some others?")

    return res.fun, res.x

def predict(theta,X,poly_degree=1):
    return sigmoid(polymap(X,degree=poly_degree) @ theta).round()

def plot_data(X,y,theta=None, xlabel="X",ylabel="Y",pos_legend="Positive",neg_legend="Negative", decision_boundary=False, poly_degree=1):
    """Simple. Assumes X has the theta0 feature in the 0th column already

    Plots `y` value (pos/neg) for features x1, x2. 

    Optionally, adds decision boundary contour specified by `theta`, calculated against range of x feature values
    mapped to higher order polynomial with order `poly_degree`
    """
    fig = plt.figure()
    plt.scatter(X[y==0,1], X[y==0,2], c='k', marker='+', linewidths=1.5, edgecolors='k', s=60, label=neg_legend)
    plt.scatter(X[y==1,1], X[y==1,2], c='g', marker='o', linewidths=1.5, edgecolors='k', s=40, label=pos_legend)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if decision_boundary:
        if theta is None:
            utils.printYellow("Must provide theta kwarg when trying to add decision boundary plot_data(X,y,theta=[],decision_boundary=True")
        else:
            ## Build decision boundary contour
            xx = np.linspace(X[:,1].min(), X[:,1].max())
            yy = np.linspace(X[:,2].min(), X[:,2].max())
            zz = np.zeros((len(xx),len(yy)))

            for ix in range(len(xx)):
                for iy in range(len(yy)):
                    #                                                       | Because of how polymap reshapes 1d arrays, must transpose before multiplying
                    zz[ix,iy] = polymap([[1, xx[ix],yy[iy]]],degree=poly_degree) @ theta
    
            c = plt.contour(xx,yy,zz,levels=[0])
            c.collections[0].set_label('Decision Boundary')
    plt.legend()
    plt.show(block=False)
    return fig

def _test_unregularized():
    ans = { "J_init":0.693147180559946, 
            "Grad_init": [-0.100000, -12.009217, -11.262842],
            "J_opt":0.203506,
            "theta_opt":[ -24.932775,0.204406,0.199616],
            "prob":0.774321,
            "accuracy":89.00
          }
    print("\n\n==============================Begin Unregularized Logistic Regression Test============================\n")
    ## Load data into dataframe
    df = pd.read_csv(os.path.join(_smarty_dir,"test","data","ex2data1.txt"),names=["exam1","exam2","admitted"])
    X = np.array(df.iloc[:,0:2])
    y = np.array(df["admitted"])

    ## Prepend the theta0 column to X, form initial theta
    X = np.insert(X, 0, 1, axis=1)
    theta_init = np.zeros(X.shape[1])

    ## Plot
    plot_data(X,y,xlabel="Exam 1 Score",ylabel="Exam 2 Score", pos_legend="Admitted",neg_legend="Not Admitted")

    ## Compute unregularized cost and gradient
    J = compute_cost(X,y,theta_init,lam=0)
    grad = compute_gradient(X,y,theta_init,lam=0)
    print("\n===Cost Function===")
    print("\tCost at initial theta (zeros): {:.5}\t\t[MATLAB: {:.5}]".format(J, ans["J_init"]))
    print("\tGrad at initial theta (zeros): {}\t\t[MATLAB: {}]\n".format(grad, ans["Grad_init"]))

    ## Solution
    J, theta = solve_regression(X,y,poly_degree=None,lam=0)
    print("\n===Optimized Solution===")
    print("\tCost at optimum theta: {:.5}\t\t[MATLAB: {:.5}]".format(J, ans["J_opt"]))
    print("\tOptimum theta: {}\t\t[MATLAB: {}]".format(theta, ans["theta_opt"]))
    plot_data(X,y,theta=theta,decision_boundary=True,poly_degree=1,xlabel="Exam 1 Score",ylabel="Exam 2 Score", pos_legend="Admitted",neg_legend="Not Admitted")

    prob = sigmoid(np.array([1,45,85]) @ theta)
    print("\tFor a student with scores 45 and 85, we predict an admission probability of {}\t\t[MATLAB: {}]".format(prob,ans["prob"]));

    ## Accuracy
    p = predict(theta,X)
    training_accuracy = (p==y).mean()*100.0
    print("\n===Training Accuracy===")
    print("\tAccuracy: {}\t\t[MATLAB: {}]".format(training_accuracy, ans["accuracy"]))
    plt.title("Training Accuracy = {:.5}".format(training_accuracy))

    print("\n\n==============================End Unregularized Logistic Regression Test============================\n")
    return X,y,J,theta,training_accuracy

def _test_regularized(lam=1, poly_degree=6):
    ###### Part 2: Regularization on more complex dataset
    ans = { "J_init":0.693147180559946, 
            "Grad_init": [0.008474576271186, 0.018788093220339, 0.000077771186441],
            "J_opt":0.529002737482998,
            "theta_opt":[1.272466148187809, 0.624959478618474, 1.180989312686644],
            "accuracy":83.050847
      }
    ## Load data into dataframe
    print("\n\n==============================Begin Regularized Logistic Regression Test============================\n")
    utils.printBlue("Note, optimized solutions won't exactly match MATLAB's due to minimization algorithm differences and ")
    df = pd.read_csv(os.path.join(_smarty_dir,"test","data","ex2data2.txt"),names=["Microchip Test 1","Microchip Test 2","PassFail"])
    X = np.array(df.iloc[:,0:2])
    y = np.array(df["PassFail"])

    ## Prepend the theta0 column to X, form initial theta
    X = np.insert(X, 0, 1, axis=1)
    theta_init = np.zeros(X.shape[1])
    
    ## Plot
    plot_data(X,y,xlabel="Microchip Test 1",ylabel="Microchip Test 2", pos_legend="Pass",neg_legend="Fail")
    
    ## Map and setup problem
    Xp = polymap(X,degree=poly_degree)
    theta_init = np.zeros(Xp.shape[1])

    ## Compute regularized cost and gradient
    J = compute_cost(Xp,y,theta_init,lam=lam)
    grad = compute_gradient(Xp,y,theta_init,lam=lam)
    print("\n===Cost Function===")
    print("\tCost at initial theta (zeros): {:.5}\t\t[MATLAB: {:.5}]".format(J, ans["J_init"]))
    print("\tGrad at initial theta (zeros)[0:3]: {}\t\t[MATLAB: {}]\n".format(grad[0:3], ans["Grad_init"]))

    ## Solution
    J, theta = solve_regression(X,y,poly_degree=poly_degree,lam=lam)
    print("\n===Optimized Solution===")
    print("\tCost at optimum theta: {:.5}\t\t[MATLAB: {:.5}]".format(J, ans["J_opt"]))
    print("\tOptimum theta[0:3]: {}\t\t[MATLAB: {}]".format(theta[0:3], ans["theta_opt"]))
    plot_data(X,y,theta=theta,decision_boundary=True,poly_degree=poly_degree,xlabel="Microchip Test 1",ylabel="Microchip Test 2", pos_legend="Pass",neg_legend="Fail")
    ## Accuracy
    p = predict(theta,Xp)
    training_accuracy = (p==y).mean()*100.0
    print("\n===Training Accuracy===")
    print("\tAccuracy: {}\t\t[MATLAB: {}]".format(training_accuracy, ans["accuracy"]))

    plt.title("Regularization lambda = {}, Poly order = {}\nAccuracy = {:.5}".format(lam,poly_degree,training_accuracy))

    print("\n\n==============================End Regularized Logistic Regression Test============================\n")
    return X,y,J,theta,training_accuracy

def test():
    """Comprehensive unit test would be nice. For now, just perform the same test procedure as MATLAB logistRetressionTest

    """
    ## Maybe this! Save everything from MATLAB verified into mat file, read and compare
    # var = sio.loadmat("test/polyTrue.mat")
    # X = var['X'] >> loads into np array X! Neat!

    ###### Example Dataset 1: Simple Exams vs Admittance (Unregularized)
    ## Define matlab solutions
    unreg_res = _test_unregularized()
    reg_res = _test_regularized()


    

if __name__ == "__main__":
    test()

    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')















