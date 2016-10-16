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
File name: kmeans.py
Created:  15/Oct/2016
Modified: 15/Oct/2016

See _SMARTY_DIR/projects/Smarty-Kmeans.ipynb for usage demonstration

Description:
    Utilities to perform kmeans unsupervised classification.  

    Algorithm:
        1. Select random points in dataset to initialize centroids to
   
         ↱  ↴   2. Associate each point in dataset to nearest centroid
     repeat ↲   3. Recompute centroid as average of all of it's assocaited datapoints
   
        4. Done! 

Note:
    Error checking, vector sizes, etc are omitted for the time being.

Nomenclature:
    See README.md

TODO/Improvements:
    * Better dedicated 'cohesion' estimate. AKA implement actual cost function()
    * Dedicated KmeansClassifier() class to handle procedural logic
    * Post-classification analysis of whole centroid set, see if we're underfitting ought to be 
        pretty easy - average cohesion for a centroid would be high if it is spanning different 
        distinct clusters. 
    * Slow. Like holy shit it slow! Mainly associate_centroids(). Can't figure out why or how to speed it up

    """

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import numpy as np
from numba import jit
import scipy.io as sio
import profile

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(_here)[0])   # Absolute version of sys.path.append(..). Helpful so we can run this module standalone...
from smartypy import utils, _SMARTY_DIR

###### Module variables

##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*
class Centroid(object):
    def __repr__(self):
        return "Centroid after {} updates with position {} - has {} associated centroids".format(len(self.positions)-1,self.x, len(self.associated))

    @property
    def x(self):
        return self.positions[-1]
    
    @property
    def cohesion(self):
        """Handy for me, probably not statistically correct"""
        return self._cohesion[-1]
    
    @property
    def associated(self):
        return self.X[self.associated_ix]

    def __init__(self, X, x):
        """Initialize centroid to n-dimensional `x` """

        ## positions -> list of all centroid positions, kept for historcal purposes
        self.positions = [x]

        ## associated -> list of associated datapoints
        self.associated_ix = []
        self._cohesion = [None]
        self.X = X
    
    def distance(self,x):
        """Compute distance of centroid to `x` datapoint
        
        Distance is computed as teh squared norm of the difference between the two points
        """
        return np.linalg.norm(self.x-x)**2.0
    
    def calc_cohesion(self):
       self._cohesion.append(np.mean([self.distance(x) for x in self.associated]))
    
    def associate(self,ix):
        """Associate datapoint x with centroid object"""
        self.associated_ix.append(ix)
    
    def update(self):
        """Update current centroid estimate as the average of all assocaited points. Clear assocaited"""
        self.calc_cohesion()
        self.positions.append(np.mean(self.associated,axis=0))
        self.associated_ix = []


##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def initialize_centroids(X,num):
    """Initialize `num` centroids to randomly selected points in the X matrix
    """
    m,n = X.shape
    rand_indices = np.random.permutation(m)

    return [Centroid(X,x) for x in X[rand_indices[0:num]]]

def associate_centroids(X,centroids):
    """Associate each datapoint (row of X) with the Centroid() closest to it
    
    Crazy slow... like way slower than MATLAB. Why is that? Tried about a dozen different 
    ways to formulate this...
    """
    for ix,x in enumerate(X):
        distances = [centroid.distance(x) for centroid in centroids]
        
        # ## Smallest distance gets the point associated! 
        centroids[np.argmin(distances)].associate(ix)
    return centroids

def update_centroids(centroids):
    '""Simple wrapper""'
    for centroid in centroids:
        centroid.update()
    
def classify_kmeans(X,num,max_iters=10,min_cohesion=0.001):
    """Wrapper function to train/classify a dataset
    
    Kwargs:
        min_cohesion:   (float) if overall cohesion (avg dist between centroid and their associated datapoints)
                        is less than this, break out of loop early
        
    """
    centroids = initialize_centroids(X,num)
    prev_cohesion = 1e6
    print("K-means algo grouping dataset into {} features, with maximum iterations {}\n".format(num,max_iters))
    for ii in range(max_iters):
        print("\rClassification iteration {}, %{:3.2f} Finished with average cohesion {:3.3}".format(ii,float(ii/max_iters)*100,prev_cohesion), end="")
        centroids = associate_centroids(X,centroids)
        
        ## Dinky check for convergence. Pretty dumb and not well thought out. 
        if ii > 0:
            overall_cohesion = np.mean([c.cohesion for c in centroids])
            delta = abs(prev_cohesion-overall_cohesion)
            if min_cohesion is not None:
                if delta < min_cohesion:
                    print("\nCentroid cohesion is no longer changing. Stopping classification")
                    break
            prev_cohesion = overall_cohesion
    
        if ii < max_iters-1:
            update_centroids(centroids)
    print("\nClassification finished with overall average cohesion {:3.3f}".format(overall_cohesion))
    return centroids

            
    




#--------------------------------------------------------------------------------
#------------ Not portable or general purpose: test specific functions ----------
#--------------------------------------------------------------------------------
def _load_sample_data():
    ###### Load up sample data
    dataset = os.path.join(_SMARTY_DIR,"test","data","ex7data2.mat")
    mat = sio.loadmat(dataset)
    X = mat['X']
    return X

def test(clusters=3):
    """Pretty dumb and not production ready. Just calls the other _test() functions that demonstrate module use"""
    X = _load_sample_data()
    m,n = X.shape
    print("Testing unsupervised classification of dataset ({} x {}) by grouping it into {} clusters".format(m,n,clusters))
    centroids = classify_kmeans(X,clusters)
    return X,centroids




##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    X,centroids = test()
    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')