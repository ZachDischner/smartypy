
# SmartyPy
Simple repository featuring duplicated Matlab/Octave and Python Machine Learning utilities. Made to pair with the Coursera Machine Learning introduction course taught by Andrew Ng.
https://www.coursera.org/specializations/machine-learning?utm_source=gg&utm_medium=sem&campaignid=655866957&adgroupid=34113811915&device=c&keyword=coursera%20machine%20learning&matchtype=p&network=g&devicemodel=&adpostion=1t1&creativeid=119391632947&hide_mobile_promo&gclid=CjwKEAjwmMS-BRCm5dn51JLbp1wSJACc61tF1ufsQo0cWmXtoBUISADwzRp9PQf8vZAxP1N3APWpTRoC7-vw_wcB

### Intent and Audience
This is predomenantly intended to be a learning exercise. Not at upstaging existing and sophisticated machine learning libraries

### Notation Convention
    n:      Number of features (excluding x0 feature)
    m:      Number of examples/samples
    x:      Feature vector dataset (m x 1)
    X:      Feature or Design Matrix (m x n+1)
    Xn:     Normalized Feature Matrix (m x n+1)
    y:      Target/Solution vector (m x 1)
    J:      Cost of a sample (single value)
    theta:  Linear Regression Coefficient Vector (n+1 x 1)
    h:      Hypothesis of form: h(theta) = X @ theta

## Setup

### Python
Anaconda's Python distribution is recomended for getting started. Easiest way to get started is to install and create a new Python 3.5 environment from the included (not minimal) `py-environment.yml` file. 
http://conda.pydata.org/docs/using/envs.html

```
$ conda env create -f py-environment.yml
```
Otherwise, if you don't have Anaconda, a working Python 3.5+ environment (yes, you need 3.5+) and a few ancilary modules is all you need. 

### Clone And Run


        
