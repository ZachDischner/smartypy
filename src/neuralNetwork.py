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

Nomenclature:
    See README.md ==> Notation Convention section for an explanation of variables. 

Examples:


TODO:
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
from scipy.optimize import minimize, fmin_cg

## Local utility module
_here = os.path.dirname(os.path.realpath(__file__))
_smarty_dir =  os.path.split(_here)[0]     # Always .. from smarty files
sys.path.insert(0, _here)
import utils

###### Module variables

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*

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


    
##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    test()

    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')