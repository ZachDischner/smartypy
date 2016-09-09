import os
import sys

## Really temporary file so we can import everything in src under the 
# smartypy name. Should figure out proper package structure and whatnot 
# instead... 

_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append( os.path.join(_here, "src"))
from src import *