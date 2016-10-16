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
File name: PrincipleComponentAnalysis.py
Created:  15/Oct/2016
Modified: 15/Oct/2016

Production ready? Lol no.

See _SMARTY_DIR/projects/Smarty-PCA.ipynb for usage demonstration
 

Description:
    Utilities to perform Principle Component Analysis and subspacification 

Nomenclature:
    See README.md

TODO/Improvements:
"""

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import numpy as np
from scipy.linalg import norm
from numba import jit
import scipy.io as sio
import profile
import pylab as plt
## Also get some because I don't like typing plt.this and plt.that all the time
from pylab import plot,figure,legend,show,grid,scatter,axes,subplot,title

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(_here)[0])   # Absolute version of sys.path.append(..). Helpful so we can run this module standalone...
from smartypy import utils, _SMARTY_DIR
from smartypy.utils import is_normalized, normalize_features, denormalize

###### Module variables

##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*
class PCApprox(object):
    """Class that wraps up a dataset and applies PCA to iter
    1. Initialize/load 
    2. analyze() - pulls out covariance and eigenstuff
    3. reduce() to lower dimensional space
        now PCApprox.Z is a lower dimensional PCA rep of the input xrange
    """
    def __init__(self,X,confidence=0.95):
        self.X = X
        self.m, self.n = X.shape
        self.Xn, self.mu, self.sig = normalize_features(X)
        self.analyze()
    
    def analyze(self):
        ## Calculate covariance and eigenvectors
        self.cov = 1.0/self.m * self.X.T @ self.X
        self.U, self.S, self.V = np.linalg.svd(self.cov)
    
    def reduce(self,k=None):
        if k is None:
            k = self.determine_min_features()

        self.U_reduce = self.U[:,0:k]

        ## Map X to lower dimensions
        self.Z = self.project(self.Xn)
        return self.Z
    
    def project(self,x):
        """Project x from (n) dimensional space onto (k) lower dimensional space. 
        
        Need to run reduce() first on main X dataset"""
        return np.atleast_2d(x) @ self.U_reduce

    def approximate(self,x=None):
        """Recover full dimensional approximation of PCA'd dataset. 
        
        If no arguments provided, this returns an the approximation of itself, re projected
        from k to n dimensions.
        """
        if x is None:
            x = self.Z
        return np.atleast_2d(x) @ self.U_reduce.T
    
    def cost(self):
        return sum(list(map(lambda x: norm(x)**2.0, self.X-self.approximate())))/self.m
         
    
    def determine_min_features(self):
        """Nothing yet"""
        return 3
        

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*


#--------------------------------------------------------------------------------
#------------ Not portable or general purpose: test specific functions ----------
#--------------------------------------------------------------------------------   
def _load_sample_data():
    ###### Load up sample data
    dataset = os.path.join(_SMARTY_DIR,"test","data","ex7data1.mat")
    mat = sio.loadmat(dataset)
    X = mat['X']

    dataset = os.path.join(_SMARTY_DIR,"test","data","ex7faces.mat")
    mat = sio.loadmat(dataset)
    Xfaces = mat['X']

    return X, Xfaces

def test_face_decomposition(k=100):
    ###### Load and display a bunch of faces
    from smartypy.logisticRegression import displayData
    X,Xfaces = _load_sample_data()
    Xn,mu,sigma = normalize_features(Xfaces)

    ###### Perform PCA and visualize
    Xpca = PCApprox(Xn)
    figure()
    print("Visualizing the first 36 eigenvectors")
    displayData(Xpca.U[:,:35].T)
    plt.title("Top 36 Eigenvectors found in PCA of face datasets")

    ## Reduce and visualize
    figure(); subplot(1,2,1)
    displayData(Xn[:100])
    title("Original faces")
    print("Reducing face data from 1024 dimensional space to just {} what does the reconstructed dataset look like?".format(k))
    Xpca.reduce(k)
    print("Reduced space face data has normalized cost: {:3.5f}".format(Xpca.cost()))
    Xapprox = Xpca.approximate()
    subplot(1,2,2)
    displayData(Xapprox[:100])
    title("Faces reconstructed after \n(1024->{}) element PCA approximation.\nNot bad eh?!".format(k))

    return Xpca

def test_2d_vis():
    ###### Load and scatter plot
    X,Xfaces = _load_sample_data()
    print("Plotting scatter of raw data")
    figure()
    scatter(X[:,0],X[:,1],facecolors='none',edgecolors='b',label="True datapoints")
    grid()
    Xn,mu,sigma = normalize_features(X)

    ## Draw eigenvectors
    print("Performing PCA to approximate data, projecting 2d - 1d")
    Xpca = PCApprox(Xn)
    U,S,V = Xpca.U, Xpca.S, Xpca.V
    evec1 = denormalize(S[0] * U[:,0], mu,sigma)
    evec2 = denormalize(S[1] * U[:,1], mu,sigma)
    plot(*zip(mu,evec1),'k',lw=2,label="Vector of maximum variance 1")
    plot(*zip(mu,evec2),'k',lw=2,label="Vector of maximum variance 2")
    axes().set_aspect('equal', 'datalim')

    ###### Perform and visualize PCA
    ## Reduce to 1 dimension,
    Xpca.reduce(k=1)
    
    ## Approximate full dimensional data from projection
    print("Re-approximating original 2d dataset using 1d PCA projection has normalized cost {:5.5f}".format(Xpca.cost()))
    Xapprox = denormalize(Xpca.approximate(), mu, sigma)
    scatter(Xapprox[:,0],Xapprox[:,1],facecolors='none',edgecolors='r',label='Re-projected Approximate')
    
    ## Draw the loss from approximation
    label = None
    for ix in range(len(Xapprox)):
        if ix == len(Xapprox)-1:
            label="Error lost due to projection"
        plot(*zip(Xapprox[ix],X[ix]),'--k',lw=1,label=label)
    
    legend(loc='best',prop={'size':10})
    return Xpca


def test():
    """Mimics the Matlab course exercise 7 for PCA"""
    Xpca2d = test_2d_vis()
    XpcaFaces = test_face_decomposition()
    return Xpca2d, XpcaFaces
    



##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    X = test()
    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("\n\n\t>>>Testing finished! Press Any Key to Exit")
    os.system('read')