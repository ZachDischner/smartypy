import os
import sys
import numpy as np

## Really temporary file so we can import everything in src under the 
# smartypy name. Should figure out proper package structure and whatnot 
# instead... 
np.set_printoptions(suppress=False)
np.set_printoptions(precision=5)

_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append( os.path.join(_here, "src"))
from src import *