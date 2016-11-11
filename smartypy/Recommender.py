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
File name: Recommender.py
Created:  22/Oct/2016
Modified: 22/Oct/2016

See _SMARTY_DIR/projects/Smarty-Kmeans.ipynb for usage demonstration

Description:

Note:
    Error checking, vector sizes, etc are omitted for the time being.

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
    pass




##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    X = test()
    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')