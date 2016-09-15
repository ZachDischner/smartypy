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
Modified: 15/Sept/2016

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
    Error checking, vector sizes, etc are omitted for the time being. I also keep flip flopping between
    being clever and pythonic, or just keeping it simple (coming from Matlab/Octave) and doing everything
    through straight indexing. Neither sits particularly well :-\

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

Examples:
    See _test_unregularized() and _test_regularized() for good demonstration of utilities. 

    You can experiment with poly order and lambda regularization parameter on sample datasets to get the 
    best fit to our dataset

    ## Run agains test dataset
    X,y,J,theta,training_accuracy = _test_regularized(lam=0.1, poly_degree=2)

    ## Obtain a guess for new sample in same dataset
    passfail = predict_sample(theta, np.array([[1,3,0]]),poly_degree=2)
        0 

TODO:
    * Finish consolodating one-vs-all functionality into Classes to keep track of setup.
        * If happy with that setup, remove all polymap() calls from functions where they don't need to be, rely
        instead on the classes/user to pass in correct arguments much cleaner that way.
    * Smarter train_regression() function, auto pick regression cost minimization method based of size/other criteria?
    * Type/vector size error handling?
    * Optimizations, @njit,
    * Unit tests!! *doctest for small functions? pytest for bigger ones
    * I wonder if things get clear with dataframes? I.E. df=pd.DataFrame(X) is a bit easier at seeing and applying rows/columns

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
from numba import jit, njit
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize, fmin_cg

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
_smarty_dir =  os.path.split(_here)[0]     # Always .. from smarty files
sys.path.insert(0, _here)
import utils
from linearRegression import normalize_features

###### Module variables

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
@njit
def sigmoid(z):
    """ Compute sigmoid function on z. 

    Examples:
    >>> sigmoid(0)
    0.5
    """
    return 1/(1+np.exp(-z))

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


# jit dectoration speeds up nothing. 
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
    y = _dimensionalize_y(y)

    m = len(y)
    n = len(theta)
    hypothesis = hypothesize(X,theta)
    
    # This part subustitute the theta0 term with zero in the theta array since it does not get regularized
    tmp = theta[0]
    theta[0] = 0.0

    ## Cost function
    #                |THIS, -y is different than -1.0*y if y happens to be uint           |regularization
    # J = (1.0/m) * ((-1.0*y).dot(np.log(hypothesis)) - (1-y).dot(np.log(1-hypothesis))) + (lam/2.0/m)*sum(theta**2.0)
    J = (1.0/m) * sum((-1.0)*y*np.log(hypothesis) - (1-y)*np.log(1-hypothesis)) + (lam/2.0/m)*sum(theta**2.0)
    
    ## Put original theta value back. Referenced theta in the calling function would be altered otherwise
    theta[0] = tmp
    return J


def compute_gradient(X,y,theta,lam=1.0):
    """Compute Regularized gradient of cost function. See compute_cost for argument details"""
    
    ## Gradient of cost function
    y = _dimensionalize_y(y)
    m = len(y)
    n = len(theta)
    hypothesis = hypothesize(X,theta)
    # grad = []
    # for jj in range(n):
    #     grad.append( (1.0/m) * sum((hypothesis - y)*X[:,jj]))

    # This part subustitute the theta0 term with zero in the theta array since it does not get regularized
    tmp = theta[0]
    theta[0] = 0

    ## Regularization bit. 
    #                            | This part subustitute the theta0 term with zero in the theta array since it does not get regularized
    # grad += (lam/m) * np.insert(theta[1:],0,0)
    grad = (1.0/m) * X.T @ (hypothesis - y) + (lam/m) * theta;
    
    ## Put original theta value back. Referenced theta in the calling function would be altered otherwise
    theta[0] = tmp
    
    return grad


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

    return Xp

def train_regression(X,y,theta=None, poly_degree=1,lam=1):  
    """Train a logistic regression problem represented by Design Matrix X and results y
    
    Prints a big red warning if minimization did not occur. Tries a few differnet methods.
    Just tries two different scipy.optimize.minimize() methodds for now. Better would be to choose
    the best option for a given problem size etc, and have a more structured way of attempting multiple
    optimizations. This function could be consolodated to a helper module and used in many ML contexts.
    See http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html for good discussion of different
    methods and their strengths and weaknessed.

    Could always write our own, or use a recommended solver from Python Coursera https://github.com/royshoo/mlsn/blob/master/python/courseraEx02.py

    Returns:
        cost:   (real) Cost at minimized value of theta
        theta:  (array reals) theta vector which minimizes cost function for the design matrix X matching to
                solution set y. 
    """

    ## Poly map the design matrix
    X = polymap(X, degree=poly_degree)
    
    if theta is None:
        theta = np.zeros(X.shape[1])

    print("Solving for minimum-cost theta using initial conditions theta = {:.15}...\n\tAnd order {} polynomial mapped X matrix".format(str(theta),poly_degree))
    
    ## Simple wrapper function to make the call to get J,cost a function with just one input.
    # I think it is cleaner than mucking up the call to minimization functions with Kwargs and whatnot
    def grad(theta):
        return compute_gradient(X,y,theta,lam=lam)
    def cost(theta):
        return compute_cost(X,y,theta,lam=lam)
    def cost_and_grad(theta):
        return cost(theta),grad(theta)
    
    ## The 'Truncated Newton' TNC method seems to work well for minmimization
    res = minimize(cost_and_grad, theta, method='TNC', jac=True)

    if res.success is True:
        utils.printGreen("Cost minimization using 'TNC' method has succeeded with minimum cost {:.5} for theta = {:.35}...".format(res.fun,str(res.x)))
        return res.fun, res.x
    else:
        utils.printYellow("Minimization using 'TNC' failed! Trying 'BFGS' method")
        res = minimize(cost_and_grad, theta, method='BFGS', jac=True)
        if res.success is True:
            utils.printGreen("Cost minimization using 'BFGS' method has succeeded with minimum cost {} for theta = {:.25}...".format(res.fun,str(res.x)))
        else:
            utils.printRed("Logistic Regression cost minimization failed using two methods. Maybe try some others?")
            return None
    ## For explicitness
    cost = res.fun
    theta = res.x
    return cost, theta

def predict_sample(theta,sample,poly_degree=1,return_prob=False):
    """Predict solution of sample using a determined theta. 

    Args:
        sample:     (vector Real) Vector of feature samples. I.E. a supposed row of X
        theta:      (Vector Real) Vector of trained fit coeficients

    kwargs: 
        poly_degree: (Real) poly_degree=1 means no higher order polynomial terms used. 
                        MUST provide the same poly_degree as you used to calculate theta!!!
        return_prob: (Bool) Return the probability that the sample belongs to the same solution 
                        set as theta describes

    Returns:
        match:  (Real) Indicator of a match. Either 0 or 1 if return_prob is False, otherwise
                    it is a floating point number between 0 and 1 (sigmoid...)
    """
    # If theta was computed with a polymapped design matrix, the sample must be mapped in the same fashion
    polymapped_sample = polymap(sample,degree=poly_degree)
    match_indicator = np.array(sigmoid( polymapped_sample @ theta))
    if return_prob:
        match = match_indicator.item()
    else:
        match = match_indicator.round()
    return match

def train_multi_classification(X,y,lam,classifications=None,poly_degree=1):
    """Trains multiple logistic regression classifiers and returns all classifiers in 
    a matrix (thetas) where the i-th row corresponds to the the theta that predicts
    the i-th classification. 

    Maybe return dataframe? 

    Args:
        X:  (Real matrix) Design matrix (with or without offset term x0)
        y:  (Real vector) Training solutions per sample (row) of X
    
    Kwargs:
        classifications:    (Real list) Unique solutions that X could represent. If not 
                            provided, form out of unique y solutions
        lam:                (Real) Lambda regularization parameter (0 for unregularized)
    
    Returns:
        classifiers: (DataFrame) Dataframe of classifier predictions where the colummns/keys
    """
    ## Get different classifications if they aren't provided
    if classifications is None:
        classifications = np.unique(y)

    classifications.sort()      # Seems like an overall good idea?
    m,n = X.shape
    
    classifiers = pd.DataFrame()
    for classification in classifications:
        #                                | one-versus-all, 1 for a positive match, 0 for all others
        cost, theta = train_regression(X,y==classification, poly_degree=poly_degree, lam=lam)
        classifiers[classification] = theta

    return classifiers

def predict_classifier(classifiers, sample, poly_degree=1):
    """Predict which classifier the sampel most likely belongs to

    Args:
        classifiers:    (DataFrame) Dataframe of classifier columns of [theta] mappings
        sample:         (Vector 1 x n) Sample array. I.E. what could be a new row of the X matrix

    Kwargs:
        poly_degree:    (Real) Same poly degree the design matrix was mapped to which
                        the `classifiers` were trained upon.

    Returns:
        prediction: (_) Most likely classification for the sample dataset
        confidence: (Real) Percent likliness that the sample is correctly classified by prediction
    """
    polymapped_sample = polymap(sample,degree=poly_degree)
    
    ## Apply `predict sample` to each column
    probabilities = classifiers.apply(lambda col: predict_sample(col.values, polymapped_sample,return_prob=True))
    prediction = probabilities.idxmax()
    confidence = 100 * probabilities[prediction]
    return prediction, confidence

##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*

class BinaryClassifier(object):
    """Supervised binary classification class. Really basic placeholder.
   
    Only works with numeric classifications.
    """
    def __repr__(self):
        description = """Binary Regression. X: ({}x{}), y: ({}), Poly mapping degree: {}
        """.format(self.X.shape[0],self.X.shape[1], len(self.y), self.degree)
        return description

    def __init__(self,X,y,poly_degree=None,hush=False):
        self.X = X
        # if 
        self.y = y.astype('int16')
        self.degree = poly_degree
        self.hush = hush

    def report(self,s):
        if self.hush:
            return
        print(s)

    def train(self):
        """Trains dataset
        """
        res = train_regression(self.X, self.y, poly_degree=self.degree)
        if res is None:
            self.report("Binary Regression Failed! Maybe check into the minimization effort, or try some higher order polynomial mapping")
            return False
        ## Success! 
        cost, self.theta = res
        p = predict_sample(self.theta,self.X,poly_degree=self.degree)
        self.accuracy = (p==self.y).mean()*100.0
        self.report("Dataset trained with accuracy against Design Matrix: {}".format(self.accuracy))
        return True

    def classify(self, sample):
        """Tell whether or not the sample is positively (1) or negatively classified (0) according to the
        dataset that this BinaryClassifier was trained upon
        """
        ## Basic vectorization
        if sample.ndim>1:
            return [self.classify(s) for s in sample]

        ###### Get prediction and probablity that the fit is good
        prediction = predict_sample(self.theta,sample,poly_degree=self.degree)
        prob = predict_sample(self.theta,sample,poly_degree=self.degree, return_prob=True)
        if prediction == 1:
            self.report("Sample is a positive match for regression with probalistic confidence %{:3.5}".format(100.0*prob))
        else:
            self.report("Sample is a negative  match for regression with probalistic confidence %{:3.5}".format(100.0*(1-prob)))
        return prediction



#------------------------------------------- -------------------------------------
#------------ Not portable or general purpose test specific functions -----------
#--------------------------------------------------------------------------------

def plot_data(X,y,theta=None, xlabel="X",ylabel="Y",pos_legend="Positive",neg_legend="Negative", decision_boundary=False, poly_degree=1):
    """Simple. Assumes X has the theta0 feature in the 0th column already

    Plots `y` value (pos/neg) for features x1, x2. 

    Optionally, adds decision boundary contour specified by `theta`, calculated against range of x feature values
    mapped to higher order polynomial with order `poly_degree`
    """
    fig = plt.figure()
    plt.scatter(X[y==0,1], X[y==0,2], c='k', marker='x', linewidths=1.5, edgecolors='k', s=60, label=neg_legend)
    plt.scatter(X[y==1,1], X[y==1,2], c='g', marker='+', linewidths=1.5, edgecolors='k', s=40, label=pos_legend)
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

def displayData(X):
    """Taken unabashedly from https://github.com/royshoo/mlsn/blob/master/python/courseraEx03.py

    I really don't like how much time I waste on fiddling with plots and whatnot. Best to take someone 
    else's and focus on the real stuff.
    """
    # python translation of displayData.m from coursera
    # For now, only "quadratic" image
    example_width = np.round(np.sqrt(X.shape[1]))
    example_height = example_width
    
    display_rows = np.floor(np.sqrt(X.shape[0]))
    display_cols = np.ceil(X.shape[0]/display_rows)

    pad = 1

    display_array = -np.ones((pad+display_rows*(example_height+pad), pad+display_cols*(example_width+pad)))

    curr_ex = 0

    for j in range(display_rows.astype(np.int16)):
        for i in range(display_cols.astype(np.int16)):
            if curr_ex == X.shape[0]:
                break
            max_val = np.max(np.abs(X[curr_ex,:]))
            rowStart = pad+j*(example_height+pad)
            colStart = pad+i*(example_width+pad)
            display_array[rowStart:rowStart+example_height, colStart:colStart+example_width] = X[curr_ex,:].reshape((example_height,example_width)).T/max_val

            curr_ex += 1
        if curr_ex == X.shape[0]:
            break

    plt.imshow(display_array,extent = [0,10,0,10],cmap = plt.cm.Greys_r)
    plt.show()

def _test_unregularized():
    """Test function matches behavior and values as in (coursera submitted and verified)
    logisticRegressionTest.m script in the `_smarty_dir`/test directory, part 1
    """
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
    J, theta = train_regression(X,y,poly_degree=None,lam=0)
    print("\n===Optimized Solution===")
    print("\tCost at optimum theta: {:.5}\t\t[MATLAB: {:.5}]".format(J, ans["J_opt"]))
    print("\tOptimum theta: {}\t\t[MATLAB: {}]".format(theta, ans["theta_opt"]))

    plot_data(X,y,theta=theta,decision_boundary=True,poly_degree=None,xlabel="Exam 1 Score",ylabel="Exam 2 Score", pos_legend="Admitted",neg_legend="Not Admitted")

    prob = sigmoid(np.array([1,45,85]) @ theta)
    print("\tFor a student with scores 45 and 85, we predict an admission probability of {}\t\t[MATLAB: {}]".format(prob,ans["prob"]));

    ## Accuracy
    p = predict_sample(theta,X)
    training_accuracy = (p==y).mean()*100.0
    print("\n===Training Accuracy===")
    print("\tAccuracy: {}\t\t[MATLAB: {}]".format(training_accuracy, ans["accuracy"]))
    plt.title("Training Accuracy = {:.5}".format(training_accuracy))

    print("\n\n==============================End Unregularized Logistic Regression Test============================\n")
    return X,y,J,theta,training_accuracy

def _test_regularized(lam=1, poly_degree=6):
    """Test function matches behavior and values as in (coursera submitted and verified)
    logisticRegressionTest.m script in the `_smarty_dir`/test directory, part 2
    """
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
    J, theta = train_regression(X,y,poly_degree=poly_degree,lam=lam)
    print("\n===Optimized Solution===")
    print("\tCost at optimum theta: {:.5}\t\t[MATLAB: {:.5}]".format(J, ans["J_opt"]))
    print("\tOptimum theta[0:3]: {}\t\t[MATLAB: {}]".format(theta[0:3], ans["theta_opt"]))
    plot_data(X,y,theta=theta,decision_boundary=True,poly_degree=poly_degree,xlabel="Microchip Test 1",ylabel="Microchip Test 2", pos_legend="Pass",neg_legend="Fail")
    ## Accuracy
    p = predict_sample(theta,Xp)
    training_accuracy = (p==y).mean()*100.0
    print("\n===Training Accuracy===")
    print("\tAccuracy: {}\t\t[MATLAB: {}]".format(training_accuracy, ans["accuracy"]))

    plt.title("Regularization lambda = {}, Poly order = {}\nAccuracy = {:.5}".format(lam,poly_degree,training_accuracy))

    print("\n\n==============================End Regularized Logistic Regression Test============================\n")
    return X,y,J,theta,training_accuracy

def _test_multi(lam=1.0):
    """Test function matches behavior and values as in (coursera submitted and verified)
    multiLogisticRegressionTest.m script in the `_smarty_dir`/test directory
    """
    print("\n\n===============Begin Regularized Multi Classification Logistic Regression Test====================\n")

    ###### Load dataset
    dataset = os.path.join(_smarty_dir,"test","data","ex3data1.mat")
    if os.path.exists(dataset):
        mat = sio.loadmat(dataset)
    else:
        utils.printRed("Multi regression dataset not found! It is not tracked in git, so you'll have to download it and put it here yourself: test/data/ex3data1.mat.")
        print("Contact maintainer if you want this dataset and don't have access to get it\nNot performing test case")
        return None,None,None
    X = mat['X']
    y = mat['y'][:,0].astype('int16') # Don't want to make this a 2d array...

    # This is Python! 0 indexed! Data was provided so that '0' in the image was actually 10 in the result set
    # so that we could do some easier syntactic stuff in Matlab. No need here!
    y[y==10] = 0  

    ## Some problem definitions
    input_layer_size = 400    # 20x20 images
    num_classifications = 10  # classify images as 0-9
    m,n = X.shape
    theta_init = np.zeros(n)

    ## Display some images
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[0:100],:]
    plt.figure()
    displayData(sel)

    ## Train multiple classifiers
    classifiers = train_multi_classification(X,y,lam)

    ## Display some images with our prediction
    rand_indices = np.random.permutation(m)
    for ii in range(5):
        sel = X[rand_indices[ii],:]
        plt.figure()
        plt.imshow(sel.reshape(20,20).T, cmap=plt.cm.Greys_r)
        pred,prob =  predict_classifier(classifiers, sel)
        truth = y[rand_indices[ii]]
        plt.title("Trained Classification: {} (%{} confidence)\nTruth: {}".format(pred, int(prob),truth))

    return X,y,theta_init

    print("\n\n================End Regularized Multi Classification Logistic Regression Test=====s================\n")

def _test_BinaryClassifier():
    """Quick functional test and usage demo"""
    print("\n\n==============================Begin BinaryClassification Class Test============================\n")

    df = pd.read_csv(os.path.join(_smarty_dir,"test","data","ex2data1.txt"),names=["exam1","exam2","admitted"])
    X = np.array(df.iloc[:,0:2])
    y = np.array(df["admitted"])

    ## Create a BinaryClassifier object and train
    bp0 = BinaryClassifier(X,y,poly_degree=None)
    bp0.train()
    print("\nBinaryClassifier trained on ex2data1.test sample test score dataset. No polynomial mapping. accuracy: {:5.5}".format(bp0.accuracy))

    ## Create a BinaryClassifier object and train
    bp3 = BinaryClassifier(X,y,poly_degree=3)
    bp3.train()
    print("\nBinaryClassifier trained on ex2data1.test sample test score dataset. Order 3 polynomial mapping. accuracy: {:5.5}".format(bp3.accuracy))

    ## Let's compare
    sample = X[14]
    truth = y[14]
    guess0 = bp0.classify(sample)
    guess3 = bp3.classify(sample)
    print("\n\nFor comparision, truth value of sample 14: {}\n\tNo Poly Mapping prediction: {}\n\t3rd Order Poly Mapping Prediction: {}".format(truth, guess0,guess3))
    
    print("\n\n==============================End BinaryClassification Class Test============================\n")
    return bp0,bp3

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
    multi_res = _test_multi()
    bc_res = _test_BinaryClassifier()


    

if __name__ == "__main__":
    test()

    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')















