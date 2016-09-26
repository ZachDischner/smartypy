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
    Forward and back propegate kinda messy. Think up a recursive implementation. I like that much better
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

## I'm tired of seeing numpy nomenclature everywhere. Looks more Matlab like this way.
from numpy import log,zeros,ones

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
from smartypy import utils, _SMARTY_DIR
from smartypy.logisticRegression import sigmoid, sigmoid_prime, hypothesize, _dimensionalize_y # consolodate these to some other function?

###### Module variables
term_height, term_width = os.popen('stty size', 'r').read().split()
np.set_printoptions(precision=5,linewidth=int(term_width)-5)


##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*
class Layer(object):
    """Represents a layer in a neural network. Simply wrapped to provide easier and
    more efficient access to layer data

    Should we copy datastructures? Memory hog vs assuring data integrety?
    """
    def __repr__(self):
        me = "Neural Network Layer: Input Nodes ==> {}, output Nodes ==> {}".format(self.num_input, self.num_output)
        return me

    ## Making a property, thinking maybe we can be lazy about performing computations on input
    # data.
    @property
    def output_nodes(self):
        if self._output_nodes is None:
            print("Layer has not been activated. Call Layer.activate(X) with proper input nodes")
        return self._output_nodes

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, the_theta):
        if the_theta.ndim == 1:
            the_theta = np.array([the_theta])
        self._theta = the_theta
        self.num_input = self.theta.shape[1]-1
        self.num_output = self.theta.shape[0]
    
    def __init__(self,theta):
        """Initialize with weight matrix theta
        """
        self.theta = theta
        self.input_nodes = None
        self._output_nodes = None
    
    def activate(self, input_nodes):
        """Activate input_nodes"""
        self.input_nodes = input_nodes
        self._output_nodes = activate(self.input_nodes, self.theta)
        return self._output_nodes 

class NeuralNetwork(object):
    """Object containing multiple layers, which which you can compute the traversal
    of input dataset through many hidden activation layers
    """
    def __repr__(self):
        if len(self.layers) > 0:
            me = """Neural Network with {} Layers, mapping {} inputs to {} outputs""".format(len(self.layers), self.layers[0].num_input, self.layers[-1].num_output)
        else:
            me = """Empty Neural Network (does nothing to input)"""
        return me

    def __init__(self,thetas=None):
        """Not bothering with much until we get further in the class"""
        self.layers = []

    def add_layer(self,theta=None,layer=None):
        if layer is not None:
            self.layers.append(layer)
        if theta is not None:
            self.layers.append(Layer(theta))

    def process(self, X):
        """Process input X through the neural network. 

        Returns rows of sigmoid probabilities (final activation layer) of each row of X which
        correspond to a classification set, where the index of the maximum probability indicates
        the most likely classification of that sample.
        """
        a = X
        for layer in self.layers:
            print("processing layer {}".format(layer))
            a = layer.activate(a)

        return a


##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def activate(a,theta):
    """Process activation matrix `a`(j-1) through next layer with weights vector characterized
    by theta.

    a(j+1) = sigmoid( theta @ a(j-1))

    Args:
        theta:  (Array Reals) Weights Matrix (Includes weight of bias node)
        a:      (Array Reals) Activation Matrix with or without bias column (ones, auto detects and adds if it isn't there). 

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
    if a.ndim == 1:
        if a[0] != 1:
            a = np.insert(a,0,1,axis=0)
    else:
        if not (a[:,0]==1).all():
            a = np.insert(a,0,1,axis=1) 

    z = a @ theta.T
    a_next = sigmoid(z)
    return a_next,z

def forward_propegate(X, thetas):
    """Forward propegate examples matrix through neural network described by `thetas`. 
    
    Basically, for each layer in the network, compute forward propegation by calling the activate() function on them:
        z2 = Theta1 @ a1
        a2 = [[1],[sigmoid(z2)]
        z3 = ...

    Args:
        X:      (ndarray Reals) Design Matrix 
        thetas:  (list of vector Reals) List of Theta activation matrices, one per layer. 
    
    Returns:
        a:  (list of vector Reals) List of computed node matrices [ [a1], [a2], [a3] ... [al]] where
                a1 represents the input layer (X)
                al represents the output layer (hypothesis)
                a2-a(l-1) are the hidden layers 
        z:  (list of vector reals) List of z matrices used to transition between layers
    """
    a = []
    z = []

    ## Add 1s, offset column to X
    a.append(np.insert(X,0,1,axis=1))
    for layernum, theta in enumerate(thetas):
        a_next,z_next = activate(a[-1],theta)

        ## If on the last layer, don't add bias column.
        if layernum == len(thetas)-1:
            a.append(a_next)
        else:
            a.append(np.insert(a_next,0,1,axis=1))
        z.append(z_next)

    return a,z

def back_propegate(as_,zs,y_classifications,thetas,lam=1.0):
    """Back propegate a neural network with layer nodes in list `as_` to compute cost function gradient
    
    AKA for 3 layer:
        del3 = a3 - ys
        del2 = Theta2*del3v .* sigmoidGradient(z2)
        Delta2 = del3v*a2'
        Delta1 = del2v*a1'

        So with 3 layers, we compute [delta2,delta3] ==> [Delta1,Delta2]

    Inputs:
        Same as outputs in `forward_propegate`
    
    """
    m = len(y_classifications)
    deltas = [as_[-1]-y_classifications]
    Deltas = []
    num_layers = len(as_)

    ## In NN nomenclature, layeridx = layernumber - 1
    for layeridx in reversed(range(2,num_layers)):
        #                            | Don't BP bias term
        deltas = [deltas[-1] @ thetas[layeridx-1][:,1:] * sigmoid_prime(zs[layeridx-2])] + deltas

    for layeridx,delta in enumerate(deltas):
        ## Prepend next back-calculation to lists of deltas and Deltas
        Deltas.append(delta.T @ as_[layeridx])

    ## Finally compute the gradient!
    theta_grads = []
    for layeridx,Delta in enumerate(Deltas):
        ## Convenient reference
        theta = thetas[layeridx]
        tmp = theta[:,0].copy()
        theta[:,0] = 0
        theta_grads.append( (1.0/m) * Delta + (1.0*lam/m)*theta)
        theta[:,0] = tmp

    return theta_grads
        


def compute_cost(X,y,thetas,lam=1.0):
    """Compute classification cost of hypothesized `theta` against test dataset X and solutions y

    Basically forward propegates array of sample data through each Theta transformation layer until we arrive
    at the output layer, then compare against the provided `y` solution.

    Assumptions:
        Basic construction for now. y is the vector of solution classifications. We assume that this has the
        ENTIRE solution set in unique values, so as to construct a matrix of solution vectors (`classifications`).
        I.E if 
            y=[0,1,2]
        the classifications matrix will be a matrix such that the y value is the index of the (1) in a vector of
        possible classications:
            classifications = [[1,0,0],[0,1,0],[0,0,1]]
    
    Description:
        Cost is calculated as:
            -log(h(x))      if y == 1
            -log(1-h(x))    if y == 0

        Regularization of cost function occurs for any lambda > 0

    Args:
        X:      (ndarray Reals) Design Matrix 
        y:      (vector Reals) Solution vector
        thetas:  (list of vector Reals) List of Theta activation matrices, one per layer. 
    
    Kwargs:
        lam:    (Real >= 0) Lambda "regularization parameter"

    Returns:
        J:      (Real) Cost of hypothesis
        grad:   ([m x 1] vector Reals) Gradient of J wrt thet. [d(J)/d(theta_j)]
    """

    # y = _dimensionalize_y(y)
    m = len(y)

    ###### Perform forward propegation
    a_mats,zs = forward_propegate(X,thetas)
    hypothesis = a_mats[-1]

    ###### Form vectors of y solutions, where the 1 in each y_classification is the value of the corresponding y
    num_classifications = len(np.unique(y))
    y_classifications = zeros((m,num_classifications))
    for idx in range(m):
        y_classifications[idx,y[idx]] = 1

    ## Cost function
    #                |THIS, -y is different than -1.0*y if y happens to be uint           |regularization
    # J = (1.0/m) * ((-1.0*y).dot(np.log(hypothesis)) - (1-y).dot(np.log(1-hypothesis))) + (lam/2.0/m)*sum(theta**2.0)
    J = (1.0/m) * sum(sum((-1.0)*y_classifications*log(hypothesis) - (1-y_classifications)*log(1-hypothesis)))

    ## Regularization part - don't include regulariation of bias term (first column)
    regularization = (lam/2.0/m) * sum([(theta[:,1:]**2).sum() for theta in thetas])   # verified!
    J += regularization
    
    return J

#--------------------------------------------------------------------------------
#------------- Not portable: general purpose test specific functions ------------
#--------------------------------------------------------------------------------
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
    m,n = X.shape


    ###### Form a Neural Network
    a0,z0 = activate(X,theta1)
    a1,z1 = activate(a0,theta2)
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

    rand_indices = np.random.permutation(m)
    for ii in range(5):
        sel = X[rand_indices[ii],:]
        plt.figure()
        plt.imshow(sel.reshape(20,20).T, cmap=plt.cm.Greys_r)
        idx = hypothesis[rand_indices[ii]].argmax()
        pred = classifiers[idx]
        prob = hypothesis[rand_indices[ii]][idx]*100.0
        truth = y[rand_indices[ii]]
        plt.title("Trained Classification: {} (%{} confidence)\nTruth: {}".format(pred, int(prob),truth))

    print("\n\n==============================End Neural Network Prediction Test============================\n")
    return X,y,theta1,theta2,hypothesis,predictions

def _test_nn():
    weights = os.path.join(_SMARTY_DIR,"test","data","ex3weights.mat")
    dataset = os.path.join(_SMARTY_DIR,"test","data","ex3data1.mat")
    inits = os.path.join(_SMARTY_DIR,"test","inits.mat")
    mat = sio.loadmat(dataset)
    wmat = sio.loadmat(weights)
    inits =sio.loadmat(inits)

    X = mat['X']
    # Load y specifically into 1d array with datatype int16 (from mat file, it is an uint8. Which can cause tricksy issues)
    y = mat['y'][:,0].astype('int16')
    y[y==10] = 0
    theta1 = inits['Theta1']
    theta2 = inits['Theta2']
    thetas = [theta1,theta2]




##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    test()

    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')
