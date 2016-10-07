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
    *Forward and back propegate kinda messy. Think up a recursive implementation. I like that much better
    *Refactor NeuralNetwork() class a bit so that FP/BP since the class can know about start/stop conditions, and use 
        Layer() objects. Leave standalone functions alone. Should make the process much clearer. 
    *Implement NeuralNetwork() train method that doesn't just unnecessarily call the standalone function?
    *Using numpy Matrix instead of arrays sounds smarter. More Matlabl like where rows/columns matter. Refactor?
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
import itertools
from numba import jit, njit,float64
from scipy.optimize import minimize, fmin_cg

import multiprocessing


## I'm tired of seeing numpy nomenclature everywhere. Looks more Matlab like this way.
from numpy import log,zeros,ones

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(_here)[0])   # Absolute version of sys.path.append(..). Helpful so we can run this module standalone...
from smartypy import utils, _SMARTY_DIR
from smartypy.utils import sigmoid, sigmoid_prime, hypothesize
from smartypy.logisticRegression import displayData 

###### Module variables

## Set output for numpy display to be the terminal width
try:
    term_height, term_width = os.popen('stty size', 'r').read().split()
    np.set_printoptions(precision=5,linewidth=int(term_width)-5)
except:
    ## Doesn't work from ipython notebooks. But that's okay
    pass


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
    
    def forward_propegate(self, input_nodes):
        """Activate input_nodes"""
        self.input_nodes = input_nodes
        self._output_nodes,self.z_output = activate(self.input_nodes, self.theta)
        return self._output_nodes,self.z_output

class SupervisedNeuralNetwork(object):
    """Object containing multiple layers, which which you can compute the traversal
    of input dataset through many hidden activation layers

    Uses separate, raw function versions of forward/back propegation, training, etc all defined
    below for computation. I'd like to see a refactor where the forward/back prop is contained 
    within the class and takes advantage of some of the information that the class has.
    """

    @property
    def input_layer(self):
        return self.layers[0]
    @property
    def output_layer(self):
        return self.layers[-1]

    def __repr__(self):
        if len(self.layers) > 0:
            me = """Neural Network with {} Layers, mapping {} inputs to {} outputs. Current approx cost {}""".format(len(self.layers), self.layers[0], self.layers[-1],self.J)
        else:
            me = """Empty Neural Network (does nothing to input)"""
        return me

    def __init__(self,X,y,hidden_layer_sizes=None, num_classifications=None, regularization=1.0, train_percent=100):
        """Not bothering with much until we get further in the class
        
        If num_classifications is not provided, it is assumed that y has the full set of representative solutions
        Vectors of classiciations are built based off of y, where the classification index corresponds to the value of y
        AKA if y has 10 unique elements, y_classifications will be made of 10 element vectors, where y==0 corresponds
        to y_classifications[0] = 1

        Kwargs:
            hidden_layer_sizes:     (list) list of hidden layer node sizes. If not provided, neural network 
                                    is a simple input-output 2 layer mapping
        """
        self.train_percent = train_percent
        self.X = X
        self.m,self.n = X.shape
        self.y = y
        if num_classifications is None:
            num_classifications = len(np.unique(y))
        
        self.y_classifications = zeros((self.m,num_classifications))
        for idx in range(self.m):
            self.y_classifications[idx,y[idx]] = 1

        self.layers = [self.n, num_classifications]
        self.as_ = None
        self.zs = None
        self.lam = regularization
        self.initialize()

        if hidden_layer_sizes is not None:
            for layersize in hidden_layer_sizes:
                self.add_layer(layersize)

    def add_layer(self,num_nodes):
        """Adds a hidden layer to the neural network"""
        self.layers = self.layers[:-1] + [num_nodes] + [self.layers[-1]]
        self.initialize() 

    def initialize(self):
        def pairwise(iterable):
            """s -> (s0,s1), (s1,s2), (s2, s3), ...
            http://stackoverflow.com/questions/5434891/iterate-a-list-as-pair-current-next-in-python
            """
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)
        self.thetas = []
        
        ###### Initialize all theta mapping matrices
        for nodes_in,nodes_out in pairwise(self.layers):
            self.thetas.append(initialize_theta(nodes_in,nodes_out))

        ###### Initialize the training and cross validation matrices
        ## in work, not used yet
        rand_indices = np.random.permutation(self.m)
        num_to_train = int(self.train_percent/100*self.m)
        self.Xtrain = self.X[rand_indices[0:num_to_train]]
        self.J = self.cost()


    def forward_propegate(self):
        """Process input X through the neural network. 

        Returns rows of sigmoid probabilities (final activation layer) of each row of X which
        correspond to a classification set, where the index of the maximum probability indicates
        the most likely classification of that sample.
        """
        self.as_, self.zs = forward_propegate(self.Xtrain,self.thetas)
        hypothesis = self.as_[-1]
        return hypothesis

    def back_propegate(self):
        # Check for forward prop first...
        self.gradients = back_propegate(self.as_,self.zs,self.y_classifications,self.thetas)
        return self.gradients

    def grad(self):
        return self.back_propegate()

    def cost(self):
        self.J = compute_cost(self.Xtrain,self.y,self.thetas,y_classifications=self.y_classifications,lam=1.0,grad=False)
        return self.J

    def predict(self,X):
        return predict(X,self.thetas)

    def train(self,max_iter=10):
        self.thetas,self.train_status = train_NeuralNetwork(self.X,self.y,thetas=self.thetas,y_classifications=self.y_classifications,lam=1.0,max_iter=max_iter,report=True,check_grads=False)
        self.cost()




##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def activate(a,theta,hidden_layer=False):
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

    ## Hidden layers should use the tanh() function according to http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    if hidden_layer:
        a_next = np.tanh(z)
    else:
        a_next = sigmoid(z)
    ## Should use this for internal layers...

    return a_next,z

def initialize_theta(L_in,L_out,epsilon=0.12):
    """Randomly seed a new network parameter matrix Theta.

    Adds the bias term column as well
    
    Pretty simple funciton call

    Args:
        L_in:   (int) Number of nodes of layer feeding this Theta (to the left)
        L_out:   (int) Number of nodes of layer fed by this Theta (to the right)
    """
    return np.random.rand(L_out,L_in+1)*2*epsilon-epsilon

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
    ## Ensure X is treated as a 2d vector
    X = np.atleast_2d(X)
    a = []
    z = []

    ## Add 1s, offset column to X
    a.append(np.insert(X,0,1,axis=1))
    for layernum, theta in enumerate(thetas):
        if (layernum == 0) or (layernum==len(thetas)-1):
            hidden_layer = False
        else:
            hidden_layer = True
        a_next,z_next = activate(a[-1],theta,hidden_layer=hidden_layer)

        ## If on the last layer, don't add bias column.
        if layernum == len(thetas)-1:
            a.append(a_next)
        else:
            a.append(np.insert(a_next,0,1,axis=1))
        z.append(z_next)

    return a,z

@njit
def bp_delta(delta1,theta0,z1):
    return (delta1 @ theta0) * sigmoid_prime(z1)
@njit
def bp_Delta(delta,theta,a,lam,m):
    # ttheta = theta.copy()
    # d0 = (delta @ theta) * sigmoid_prime(z)
    
    Delta = delta.T @ a
    tmp = theta[:,0].copy()
    theta[:,0]=0
    tg = 1.0/m * Delta + (1.0*lam/m)*theta
    theta[:,0]=tmp
    return Delta,tg

def fast_back_propegate(as_,zs,y_classifications,thetas,lam=1.0,full_output=False):
    """Identical to `back_propegate` except that the delta and Delta, theta gradients are calculated layer-by-layer,
    instead of first calculating all deltas, then all Deltas, then all theta gradients. 

    Hopes were that we @jnit would speed up the calc. Didn't much, but I still wanted to separate the operation out
    in anticipation of a nice neural network class, or someday using a numbapro/cuda to offload some of this to the gpu
    """
    m = len(y_classifications)
    deltas = []
    Deltas = []
    theta_grads = []
    num_layers = len(as_)

    for ix in reversed(range(1,num_layers)):
        ## Explicitness and ease of comprehension: Going from layer B to layer A backwards
        zA = zs[ix-1]
        thetaA = thetas[ix-1]
        aA = as_[ix-1]

        # print("Processing layer ",ix)
        if ix == num_layers-1:
            deltaA = as_[-1] - y_classifications
        else:
            thetaB = thetas[ix]
            deltaA = bp_delta(deltaA,thetaB[:,1:], zA)

        DeltaAB, theta_gradAB = bp_Delta(deltaA, thetaA,aA,lam,m)

        deltas = [deltaA] + deltas
        Deltas = [DeltaAB] + Deltas
        theta_grads = [theta_gradAB] + theta_grads

    if full_output:
        return theta_grads,deltas,Deltas
    return theta_grads

def back_propegate(as_,zs,y_classifications,thetas,lam=1.0):
    """Back propegate a neural network with layer nodes in list `as_` to compute cost function gradient
    
    Speedups:
        This is probably the MOST called function in neural network training. Biggest slow down. Tried a few
        basic speed up ideas (@jit), not much helps. Best we can do is probably parallelize? Better yet, 
        write our own optimizer that makes parallel calls to back_propegate for +- estiamtes? Can be pretty slow...

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
        deltas = [deltas[0] @ thetas[layeridx-1][:,1:] * sigmoid_prime(zs[layeridx-2])] + deltas

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

def compute_cost(X,y,thetas,y_classifications=None,lam=1.0,grad=False):
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
        thetas: (list of vector Reals) List of Theta activation matrices, one per layer. 
    
    Kwargs:
        lam:    (Real >= 0) Lambda "regularization parameter"

    Returns:
        J:      (Real) Cost of hypothesis
        grad:   ([m x 1] vector Reals) Gradient of J wrt thet. [d(J)/d(theta_j)]
    """
    ## Ensure X is treated as a 2d vector (hint, X[0]==>(400,) is not 2d (1,400))
    X = np.atleast_2d(X)
    m = len(y)


    ###### Perform forward propegation
    a_mats,zs = forward_propegate(X,thetas)
    hypothesis = a_mats[-1]

    ###### Form vectors of y solutions, where the 1 in each y_classification is the value of the corresponding y
    if y_classifications is None:
        num_classifications = 10#len(np.unique(y))
        y_classifications = zeros((m,num_classifications))
        for idx in range(m):
            y_classifications[idx,y[idx]] = 1

    ## Cost function
    #                |THIS, -y is different than -1.0*y if y happens to be uint           |regularization
    # J = (1.0/m) * ((-1.0*y).dot(np.log(hypothesis)) - (1-y).dot(np.log(1-hypothesis))) + (lam/2.0/m)*sum(theta**2.0)
    J = (1.0/m) * ((-1.0)*y_classifications*log(hypothesis) - (1-y_classifications)*log(1-hypothesis)).sum()

    ## Regularization part - don't include regulariation of bias term (first column)
    regularization = (lam/2.0/m) * sum([(theta[:,1:]**2).sum() for theta in thetas])   # verified!
    J += regularization

    if grad:
        gradients = fast_back_propegate(a_mats,zs,y_classifications,thetas,lam=lam)
        return J,gradients

    return J



def predict(X,thetas):
    """Predict neural output of neural network described by `thetas` with input sample X
    """
    ###### Ensure input is treated as at least a 2D vector
    ## AKA, make sure a single row of an X matrix is treated as a (1,n) vector, instead of (n,) which is just a pain in the ass
    X = np.atleast_2d(X)

    ###### Forward propegate sample data
    as_,zs = forward_propegate(X,thetas)
    ## Hypothesis is last `a` matrix  (set of nodes) in the neural network
    hypothesis = as_[-1]

    ## Index of maximum in each hypothesis directly corresponds to the hypothesis in current formulations. Neat! 
    predictions = list(zip(hypothesis.argmax(axis=1),hypothesis.max(axis=1)))

    return predictions

def flatten_arrays(thetas):
    tvec = np.array([])
    shapes = []
    for theta in thetas:
        shapes.append(theta.shape)
        tvec = np.append(tvec,theta.flatten())
    return tvec,shapes

def unflatten(thetas_vec, shapes):
    """Reconstruct 2D theta vectors from a single flattened array according to the shapes provided

    Basically undoes `flatten_arrays()`
    """
    thetas = []
    idx = 0
    for shp in shapes:
        idx_end = idx+(shp[0]*shp[1])
        ## C order reshaping? Honestly I'm not sure which to use but this handles the reconstruction properly I believe...
        theta = np.reshape(thetas_vec[idx:idx_end], shp, order='C').copy()
        thetas.append(theta)
        idx = idx_end
    return thetas

def check_gradients(X,y,thetas,lam=1.0):
    def approximate_grad(tv,epsilon=1e-4):
        approx_grads = np.empty(tv.shape)
        tvplus = tv.copy()
        tvminus = tv.copy()
        for ix in range(len(tv)):
            if ix % 100 == 0:
                print("Numerical evaluation %{}".format(ix*1.0/m), end='\r')
            ## Alter this theta
            tvplus[ix] += epsilon
            tvminus[ix] -= epsilon
            tplus = unflatten(tvplus,shapes)
            tminus = unflatten(tvminus,shapes)

            ## Compute approximate gradient 
            Jplus = compute_cost(X,y,tplus,lam=lam)
            Jminus = compute_cost(X,y,tminus,lam=lam)
            approx_grads[ix] = (Jplus - Jminus)/(2.0*epsilon)

            ## Reset
            tvplus[ix] -= epsilon
            tvminus[ix] += epsilon
        print("")
        return approx_grads

    print("Comparing numerically approximated gradients against closed form solution")
    m,n = X.shape
    if m > 100:
        utils.printRed("HIGHLY recommended that you dont' check gradients when m ({}) is > 100. Super duper slow...".format(m))
    
    thetas_vec,shapes = flatten_arrays(thetas)
    g_approx = approximate_grad(thetas_vec)

    J,gtrue = compute_cost(X,y,thetas,lam=lam,grad=True)
    gtrue_flat,s = flatten_arrays(gtrue) 

    print("------Real------|-----Approx-----")
    for ii in range(5):
        print("   {:3.4f}\t|\t{:3.4f}\t".format(gtrue_flat[ii],g_approx[ii]))
    gdiff = np.linalg.norm(g_approx-gtrue_flat)/np.linalg.norm(g_approx+gtrue_flat)
    print("Computed norm of the difference = {}".format(gdiff))
    good = False not in np.isclose(gtrue_flat,g_approx)
    if good:
        utils.printGreen("According to numpy, this means the back propegation is working!")
    else:
        utils.printRed("According to numpy, this means that the approximation and true aren't equal...")

    return good, gtrue_flat,g_approx,shapes

iternum = 0
def train_NeuralNetwork(X,y,layer_nodes=None,thetas=None, y_classifications=None,lam=1.0,max_iter=20,report=True,check_grads=False):
    """Train a neural network with sample data X, solutions y, list of node sizes
    
    Assumptions:
        Not too smart. layer_nodes's first layer should match the X column length, layer_nodes's
        last layer should match up with the classifications of y (AKA the unique elements of y)

    Description:
        `layer_nodes` will dictate the shape and size of the neural network. Theta matrices will
        be constructed from it
        layer_nodes = [3,5,2] builds a neural network like:
                
                  /---()---\
                ()----()----\
                () \--()-----()
                ()  \-()---/ ()
                     -()--/
    Args:
        X
        y
        layer_nodes:    (list of ints) Spec for number of layers and how big they are

    Kwargs:
    """
    print("Training neural network described layer nodes={}. Regularization={}".format(layer_nodes,lam))
    m,n = X.shape
    global iternum
    iternum = 0

    ###### Initialize Theta matrices
    if thetas is None:
        thetas_init = []
        for layer_idx in range(len(layer_nodes)-1):
            thetas_init.append(initialize_theta(layer_nodes[layer_idx],layer_nodes[layer_idx+1]))
    else:
        thetas_init = thetas
    
    ## Flattened, vectorized edition
    thetas_vec,theta_shapes = flatten_arrays(thetas_init) 

    ###### Form classifications for y
    if y_classifications is None:
        num_classifications = len(np.unique(y))
        y_classifications = zeros((m,num_classifications))
        for idx in range(m):
            y_classifications[idx,y[idx]] = 1

    ###### Create FP and BP wrapper functions
    # Simplifies the calling in minimization functions, as well as un-flattens input thetas
    def cost_and_grad(thetas_vec):
        thetas = unflatten(thetas_vec,theta_shapes)
        J,g_vecs = compute_cost(X,y,thetas,lam=lam,grad=True,y_classifications=y_classifications)
        return J, flatten_arrays(g_vecs)[0]

    def cost_and_grad2(nn_params):
        J,g = nnCostFunction(nn_params,layer_nodes[0], layer_nodes[1], num_classifications, X,y,lam)
        return J,np.squeeze(np.asarray(g))

    def cost(thetas_vec):
        return cost_and_grad(thetas_vec)[0]
    
    def grad(thetas_vec):
        return flatten_arrays(cost_and_grad(thetas_vec)[1])[0]

    ###### Callback function to display progress, etc
    def status_update(XX):
        global iternum
        print("Iter: {}, cost: {:3.5f}".format(iternum,cost(XX)))
        iternum += 1
        ## Can put in logic to see if the time between iterations has slowed, or if cost isn't converging anymore...
        # if iternum > 50:
        #     raise Exception("Too long!")

    ###### Perform Minimization!
    initialcost = cost(thetas_vec)
    print("Performing minimization (this could take a while). Initial cost: {}".format(initialcost))
    ## Speed/accuracy stats are based off of test case. Favorites first

    ## Newton CG: Speed = 8, minimization=2
    ret = minimize(cost_and_grad, thetas_vec, method='newton-cg', jac=True, callback=status_update, options={"maxiter":max_iter,"disp":report})

    ## L-BFGS-B: Speed = 10, minimization=4    
    # ret = minimize(cost_and_grad, thetas_vec, method='L-BFGS-B', jac=True, callback=None, options={"maxiter":max_iter,"disp":report})

    ## CG: Speed = 2, minimization=8
    # ret = minimize(cost_and_grad, thetas_vec, method='cg', jac=True, callback=status_update, options={"maxiter":max_iter,"disp":report})

    ## BFGS: Speed = 0---, minimization=5 
    # ret = minimize(cost_and_grad, thetas_vec, method='BFGS', jac=True, callback=status_update, options={"maxiter":max_iter,"disp":report})

    print("Minimization completed with final cost {}".format(ret.fun))
    thetas_opt = unflatten(ret.x,theta_shapes)
    
    ## Speed = 5, minimization=5
    # ret = fmin_cg(cost,thetas_vec,fprime=grad,maxiter=max_iter,full_output=True,callback=status_update, disp=True)
    
    # print("Minimization completed with final cost {}".format(ret[1]))
    # thetas_opt = unflatten(ret[0],theta_shapes)

    return thetas_opt,ret


#--------------------------------------------------------------------------------
#------------- Not portable: general purpose test specific functions ------------
#--------------------------------------------------------------------------------

## Taken from previous course material as a sanity check. https://github.com/royshoo/mlsn/blob/master/python/funcsEx04.py
# Pretty close to my results but not exact? My result is closer to Matlab's solution. also runs about 90x faster B-)
# nn_params=np.append(theta1.flatten(),theta2.flatten())
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    Theta1 = np.matrix(np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
    Theta2 = np.matrix(np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))
    
    a1 = np.c_[np.ones((X.shape[0],1)),X]
    a2 = np.c_[np.ones((X.shape[0],1)),sigmoid(a1*Theta1.T)]
    a3 = sigmoid(a2*Theta2.T)

    Y = np.zeros((X.shape[0],num_labels))

    for i in range(num_labels):
        for j in range(X.shape[0]):
            if y[j] == i+1: # To be consistant with matlab program
                Y[j,i%10] = 1

    J = (np.multiply(-Y,np.log(a3))-np.multiply(1-Y,np.log(1-a3))).sum().sum()/X.shape[0]
    J += lamb*(np.power(Theta1[:,1:],2).sum().sum()+np.power(Theta2[:,1:],2).sum().sum())/X.shape[0]/2

    Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
    Delta2 = np.zeros((num_labels,hidden_layer_size+1))

    for t in range(X.shape[0]):
        # 1
        a_1 = np.matrix(np.r_[np.ones(1),X[t,:].T]).T
        z_2 = np.dot(Theta1,a_1)
        a_2 = np.matrix(np.r_[np.ones((1,1)),sigmoid(z_2)])
        z_3 = np.dot(Theta2,a_2)
        a_3 = sigmoid(z_3)

        # 2
        yvec = np.zeros((num_labels,1))
        yvec[y[t]-1] = 1
        delta3 = a_3-yvec

        # 3
        delta2 = np.multiply(Theta2.T*delta3,np.matrix(np.r_[np.ones((1,1)),sigmoid_prime(z_2)]))

        # 4
        delta2 = delta2[1:]
        Delta2 += delta3*a_2.T
        Delta1 += delta2*a_1.T

    Theta1_grad = Delta1/X.shape[0]
    Theta2_grad = Delta2/X.shape[0]

    Theta1_grad[:,1:] = Theta1_grad[:,1:]+Theta1[:,1:]*lamb/X.shape[0]
    Theta2_grad[:,1:] = Theta2_grad[:,1:]+Theta2[:,1:]*lamb/X.shape[0]
    
    grad = np.r_[np.matrix(np.reshape(Theta1_grad,Theta1.shape[0]*Theta1.shape[1],order='F')).T,np.matrix(np.reshape(Theta2_grad,Theta2.shape[0]*Theta2.shape[1],order='F')).T]

    return J,grad

def _load_test_data():
    ###### Load up sample data
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

    ## Classifiers are shifted by 1 to the array index, since the theta training was done in Matlab with 1 indexing
    theta1 = wmat["Theta1"] # provided already trained weights
    theta2 = wmat["Theta2"]
    return X,y,theta1,theta2

def _test_prediction():
    """Comprehensive unit test would be nice. For now, just perform the same test procedure
    as MATLAB neuralNetworkPredictiontest.m
    """
    print("\n\n==============================Begin Neural Network Prediction Test============================\n")

    X,y,theta1,theta2 = _load_test_data()
    m,n = X.shape
    classifiers = [1,2,3,4,5,6,7,8,9,0]

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
    X,y,theta1,theta2 = _load_test_data()
    m = len(y)

    num_classifications = len(np.unique(y))
    y_classifications = zeros((m,num_classifications))
    for idx in range(m):
        y_classifications[idx,y[idx]] = 1

    ###### Train a neural network
    layer_nodes = (400,25,10)
    thetas_opt,ret = train_NeuralNetwork(X,y,layer_nodes)
    return X,y,thetas_opt,y_classifications

def _test_NeuralNetworkObject():    
    print("\n\n==============================Begin SupervisedNeuralNetwork() Test============================\n")

    X,y,theta1,theta2 = _load_test_data()

    ## Look how simple it is!
    print("snn = SupervisedNeuralNetwork(X,y,hidden_layer_sizes=(400,25,10), num_classifications=10, regularization=1.0)\nsnn.train()\n")
    snn = SupervisedNeuralNetwork(X,y,hidden_layer_sizes=(25), num_classifications=10, regularization=1.0)
    snn.train(max_iter=100)
    print("SupervisedNeuralNetwork() object trained with final cost")
    print("\n\n==============================End SupervisedNeuralNetwork() Test============================\n")
    return snn

def _learning_curves():
    """Just testing this out.

    Best, make a class that accepts any kind of learner (logistic regression, linear regression, neural network) and 
    specify correct cost() and training() functions so this process is all automagic. 
    """
    print("\n\n==============================Begin Learning Curves Test============================\n")

    X,y,theta1,theta2 = _load_test_data()

    m,n = X.shape
    
    num_classifications = len(np.unique(y))
    y_classifications = zeros((m,num_classifications))
    for idx in range(m):
        y_classifications[idx,y[idx]] = 1


    rand_indices = np.random.permutation(m)
    train_percent = 80
    num_to_train = int(train_percent/100*m)
    Xtrain = X[rand_indices[0:num_to_train]]  
    ytrain = y[rand_indices[0:num_to_train]]  

    Xcv = X[rand_indices[num_to_train:]]  
    ycv = y[rand_indices[num_to_train:]]  
    y_class_cv = y_classifications[rand_indices[num_to_train:]]


    ms = [1,2,3,5,10,50,100,200,300,400,500,600,700,1000,1500,2000,2500]
    layer_nodes = (400,25,10)
    Jtrain = []
    Jcv = []
    for mm in ms:
        Xt = Xtrain[0:mm]
        yt = ytrain[0:mm]
        yclass_train = y_classifications[rand_indices[0:mm]]
        thetas,ret = train_NeuralNetwork(Xt,yt,layer_nodes=layer_nodes,y_classifications=yclass_train,lam=1)
        Jtrain.append(compute_cost(Xt,yt,thetas,y_classifications=yclass_train,lam=1))
        Jcv.append(compute_cost(Xcv,ycv,thetas,y_classifications=y_class_cv,lam=1))

    plt.figure()
    plt.plot(ms,Jtrain,lw=3,label="Training Error")
    plt.plot(ms,Jcv,lw=3,label="Cross Validation Error")
    plt.xlabel("Number of Samples")
    plt.ylabel("Cost")
    plt.title("Learning Curve Cost Plot"),plt.grid(),plt.legend()
    plt.show(block=False)

    print("\n\n-------- Performing Regularization Validation Plot")
    lambdas = [0,0.001,0.003,0.01,0.05,0.1,0.15,0.2,0.3,1,2,2.5,3,3.5,4,5,10]

    Jtrain_val = []
    Jcv_val = []
    yclass_train = y_classifications[rand_indices[0:num_to_train]]
    for lam in lambdas:

        thetas,ret = train_NeuralNetwork(Xtrain,ytrain,layer_nodes=layer_nodes,y_classifications=yclass_train,lam=lam)
        Jtrain_val.append(compute_cost(Xtrain,ytrain,thetas,y_classifications=yclass_train,lam=0))
        Jcv_val.append(compute_cost(Xcv,ycv,thetas,y_classifications=y_class_cv,lam=0))

    plt.figure()
    plt.plot(lambdas,Jtrain_val,lw=3,label="Training Error")
    plt.plot(lambdas,Jcv_val,lw=3,label="Cross Validation Error")
    plt.xlabel("Regularization lambda")
    plt.ylabel("Cost")
    plt.title("Validation Curve Cost Plot"),plt.grid(),plt.legend()
    plt.show(block=False)



    print("\n\n==============================End Learning Curves Test============================\n")

    return Jtrain,Jcv,Jtrain_val,Jcv_val


def test():
    """Pretty dumb and not production ready. Just calls the other _test() functions that demonstrate module use"""
    _test_prediction()
    _test_nn()
    _test_NeuralNetworkObject()
    _learning_curves()




##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    test()
    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')
