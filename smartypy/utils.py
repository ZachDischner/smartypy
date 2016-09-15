#!/usr/bin/env python

__author__     = 'Zach Dischner'
__copyright__  = ""
__credits__    = ["NA"]
__license__    = "NA"
__version__    = "0.0.2"
__maintainer__ = "Zach Dischner"
__email__      = "zach.dischner@gmail.com"
__status__     = "Dev"
__doc__        ="""
File name: utils.py
Created:  04/Sept/2016
Modified: 04/Sept/2016

Houses a couple common utilities used by various scripts.

"""

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import subprocess
import re
from datetime import datetime

###### Module variables
_here               = os.path.dirname(os.path.realpath(__file__))
_VERSION = "0.0.0"

## Colors!!!
class bcolors:
    HEADER  = '\033[95m'
    OKBLUE  = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
## Quickies to get current git hashes. Thanks SO http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).replace("\n","")

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).replace("\n","")
def printColor(msg,color):
    print(color + str(msg) + bcolors.ENDC)

def printYellow(msg):
    printColor(msg,bcolors.WARNING)
def printGreen(msg):
    printColor(msg,bcolors.OKGREEN)
def printBlue(msg):
    printColor(msg, bcolors.OKBLUE)
def printRed(msg):
    printColor(msg,bcolors.FAIL)
























