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
File name: anomalyDetection.py
Created:  22/Oct/2016
Modified: 22/Oct/2016

STATUS IN WORK - Not complete or fully tested yet. 


Description:
    Simple anomaly detection suite. Should probably use a better one written by better people
    http://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_housing.html#sphx-glr-auto-examples-applications-plot-outlier-detection-housing-py


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
from numba import jit
import scipy.io as sio
from scipy import stats
from pylab import plot,figure,legend,show,grid,scatter,axes,subplot,title,contour,xlabel,ylabel,clabel
from matplotlib import mlab

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(_here)[0])   # Absolute version of sys.path.append(..). Helpful so we can run this module standalone...
from smartypy import utils, _SMARTY_DIR

###### Module variables

##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*


##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def estimate_gaussian_params(X):
    """ Estimate n dimensional gaussian parameters for  dataset X

    Returns:
        mu:     (array Reals) Array of means of each column of X
        var     (aray Reals) Array of variances (varma^2) of each column of X
    """
    mu = X.mean(axis=0)
    var = X.std(axis=0)**2.0
    return mu,var

def get_probability(x,mu,var):
    """ Compute probablity that x belongs to the X dataset that is characterised by [mu] and [sigma]

    Does multivariate gaussian estimate. Product of single gaussian estimates not put in place yet
    """
    m,n = np.atleast_2d(x).shape
    variance = np.diag(var)
    factor = 1.0/((2*np.pi)**(n/2.0) * np.linalg.det(variance)**0.5)
    p = np.array([ factor * np.exp( (-0.5) * ((xx).T @ np.linalg.inv(variance) @ (xx)).sum() ) for xx in np.atleast_2d(x-mu)])
    return p
    
def evaluate_epsilon(yval,pval,epsilon):
    predictions = (pval < epsilon).flatten()
    tp = ((predictions==1) & (yval.flatten() == 1)).sum()
    fp = ((predictions==1) & (yval.flatten() == 0)).sum()
    fn = ((predictions==0) & (yval.flatten() == 1)).sum()
    if (tp+fp) == 0:
        return 0, None
    precision = 1.0*tp/(tp+fp)
    recall = 1.0*tp/(tp+fn)
    F1 = (2.0*precision*recall)/(precision+recall)
    return F1, {"tp":tp,"fp":fp,"fn":fn,"precision":precision, "recall":recall}
    
def determine_threshold(yval,pval):
    """Determine best epsilon threshold parameter for determining anomalous data point
    
    DONT THINK THIS WORKS
    """

    F1 = 0
    epsilon = 0
    for _epsilon in np.linspace(min(pval),max(pval),1000):
        ## Compute stats
        _F1,stats = evaluate_epsilon(yval,pval,_epsilon)

        if _F1 > F1:
            F1 = _F1
            epsilon = _epsilon
            print("Better threshold found! {} ==> F1 {}".format(epsilon,F1))
    
    return epsilon, F1
         

#--------------------------------------------------------------------------------
#------------ Not portable or general purpose: test specific functions ----------
#--------------------------------------------------------------------------------
def plot_gaussian(mu,var,epsilon=0.089881552536):
    """ IDK what is happening here and why I have to divide epsilon by 3. 
    """
    X1,X2 = np.meshgrid(np.linspace(0,35),np.linspace(0,35))
    Z = get_probability(np.array([X1.flatten(),X2.flatten()]).T, mu, var)
    Z = np.reshape(Z, X1.shape)
    # Z[Z<epsilon] = 0
    # Z[Z>epsilon] = 1
    c = contour(X1, X2, Z, levels=[epsilon], lw=3, label="Gaussian Boundary")
    clabel(c, inline=1, fmt='Epsilon=%3.5e', colors='k',fontsize=8)

def _load_sample_data(visual_set=True):
    ###### Load up sample data
    if visual_set:
        dataset = os.path.join(_SMARTY_DIR,"test","data","ex8data1.mat")
    else:
        dataset = os.path.join(_SMARTY_DIR,"test","data","ex8data2.mat")

    mat = sio.loadmat(dataset)
    X = mat['X']
    Xval = mat['Xval']
    Yval = mat['yval']
    return X, Xval, Yval

def test():
    """Pretty dumb and not production ready. Just calls the other _test() functions that demonstrate module use"""
    X,Xval,Yval = _load_sample_data()
    mu,var = estimate_gaussian_params(X)
    pval = get_probability(Xval,mu,var)

    figure()
    plot(X[:,0],X[:,1],'b+',label='data'); xlabel("Latency (ms)"); ylabel("Throughput (Mb/s)")
    epsilon, F1 = determine_threshold(Yval,pval)
    print("Optimal epsilon and F1 score for sample dataset {}, {}".format(epsilon, F1))
    plot_gaussian(mu,var,epsilon=epsilon)

    ## Plot Outliers
    predictions = get_probability(X,mu, var)
    outliers = X[predictions < epsilon]
    plot(outliers[:,0],outliers[:,1],'ro',mfc=None,label='outliers');
    legend()
    grid()








##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    X = test()
    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')