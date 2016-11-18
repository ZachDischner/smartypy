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
File name: fishey.py
Created:  15/Nov/2016
Modified: 15/Nov/2016

STATUS IN WORK - Not complete or fully tested yet. 


Description:
    * Super simple start, just load up data into raw matrices and fit a basic sequential DNN

TODO/Improvements:

    """

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import exposure

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss


###### Module variables
_HERE = os.path.dirname(os.path.realpath(__file__))
pd.set_option('display.width', 1200)
_TEST_DIR = os.path.join(_HERE, "test_stg1")
_TRAIN_DIR = os.path.join(_HERE, "train")
_N_CHANNELS,_IMAGE_WIDTH,_IMAGE_HEIGHT = 1,128,128
_IMAGE_SHAPE = (_N_CHANNELS,_IMAGE_WIDTH,_IMAGE_HEIGHT)

_CATEGORIES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

## Set output for numpy display to be the terminal width
try:
    _, _TERM_WIDTH = os.popen('stty size', 'r').read().split()
    np.set_printoptions(precision=5, linewidth=int(_TERM_WIDTH)-5)
except:
    ## Doesn't work from ipython notebooks. But that's okay
    pass


##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*


##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*

#---------------------------------Data Loading--------------------------------
## Good for testing
def save_pickle(Xtrain, Xtrain_cat, Xtrain_fn, Ytrain):
    with open('data.pickle', 'wb') as f:
        pickle.dump([Xtrain, Xtrain_cat, Xtrain_fn, Ytrain], f)
def load_pickle():
    # Getting back the objects:
    if os.path.exists('data.pickle'):
        with open('data.pickle', 'rb') as f:
            Xtrain, Xtrain_cat, Xtrain_fn, Ytrain = pickle.load(f)
        return Xtrain, Xtrain_cat, Xtrain_fn, Ytrain

def prepare_resizes(imshape=_IMAGE_SHAPE):
    ###### Resize and reshape Training dataset
    for category in _CATEGORIES:
        train_files = glob.glob(os.path.join(_TRAIN_DIR, category, "*.jpg"))
        index = _CATEGORIES.index(category)
        for imnum, imname in enumerate(train_files):
            print("\rSaving and Resizing {} category training datasets. {:3.5f} percent complete".format(category, 100.0*imnum/len(train_files)), end="")
            tmp = _load_image(imname, imshape=imshape)
            new_imname = imname.replace("train","train_{}x{}".format(imshape[0],imshape[1]))
            if os.path.exists(os.path.dirname(new_imname)) is False:
                os.makedirs(os.path.dirname(new_imname))
            imsave(new_imname, tmp)
    
    ###### Now resize and save test images
    print("")
    test_files = glob.glob(os.path.join(_TEST_DIR, "*.jpg"))
    for imnum, imname in enumerate(test_files):
        print("\rLoading Testing datasets. {:3.5f} percent complete".format(100.0*imnum/len(test_files)), end="")
        tmp = _load_image(imname, imshape=imshape)
        new_imname = imname.replace("test_stg1","test_stg1_{}x{}".format(imshape[0],imshape[1]))
        if os.path.exists(os.path.dirname(new_imname)) is False:
            os.makedirs(os.path.dirname(new_imname))
        imsave(new_imname, tmp)




    

def _load_image(imgpath, imshape=None):
    """Just load an image and reshape. No fancy transformation or normalization"""
    x = imread(imgpath, as_grey=True)
    if imshape is not None:
        x = resize(x, imshape)
    return x

def load_train_data(imshape=_IMAGE_SHAPE, limit=None):
    Xtrain = []
    Xtrain_cat = []
    Xtrain_fn = []
    Ytrain = []

    ## See if we've already pre-processed a set of images
    basedir = _TRAIN_DIR + "_{}x{}".format(imshape[1], imshape[2])
    if  os.path.exists(basedir):
        print("Already created directories full of reshaped {} images! Just load those and don't re-resize originals again'".format(imshape))
        reshape = None
    else:
        basedir = _TRAIN_DIR
        reshape = imshape

    print("Loading training dataset")
    for category in _CATEGORIES:
        image_files = glob.glob(os.path.join(basedir, category, "*.jpg"))
        index = _CATEGORIES.index(category)
        for imnum, imname in enumerate(image_files):
            if limit is not None:
                if imnum >limit:
                    break
            print("\rLoading {} category training datasets. {:3.5f} percent complete".format(category, 100.0*imnum/len(image_files)), end="")
            Xtrain.append(_load_image(imname, imshape=reshape))
            Ytrain.append(index)
            Xtrain_cat.append(category)
            Xtrain_fn.append(os.path.basename(imname))
        print("")

    print("Loaded {} images for training".format(len(Xtrain)))
    return np.array(Xtrain), Xtrain_cat, Xtrain_fn, np_utils.to_categorical(Ytrain)

def load_test_data(imshape=_IMAGE_SHAPE, limit=None):
    Xtest = []
    Xtest_fn = []
    print("Loading training dataset")
    image_files = glob.glob(os.path.join(_TEST_DIR, "*.jpg"))
    for imnum, imname in enumerate(image_files):
        if limit is not None:
            if imnum >limit:
                break
        print("\rLoading Testing datasets. {:3.5f} percent complete".format(100.0*imnum/len(image_files)), end="")
        Xtest.append(_load_image(imname, imshape=imshape))
        Xtest_fn.append(os.path.basename(imname))
    print("")
    return np.array(Xtest), Xtest_fn

def preprocess(imageArray):
    nchannels,rows,cols = _N_CHANNELS,_IMAGE_WIDTH,_IMAGE_HEIGHT
    print("Preprocessing image array. Shape: {}".format(imageArray.shape))
    ## Equalize Exposure 
    print("Normalizing exposure by equalizing histograms")
    X = exposure.equalize_hist(imageArray)

    ## Reshape for tensorflow
    print("Transposing to (Nsamples, NChannels, Rows, Columns)")
    X = X.reshape((-1,nchannels,rows,cols))
    return X

def load(imshape=_IMAGE_SHAPE, limit=None):
    Xtrain, Xtrain_cat, Xtrain_fn, Ytrain = load_train_data(imshape=imshape, limit=limit)    
    Xtest, Xtest_fn = load_test_data(imshape=imshape, limit=limit)

    ## Preprocess all images
    Xtrain = preprocess(Xtrain)
    Xtest = preprocess(Xtest)

    return (Xtrain, Xtrain_cat, Xtrain_fn, Ytrain), (Xtest, Xtest_fn)

#---------------------------------Model Building-------------------------------
def make_model():
    model = Sequential()

    ## Create input layer
    model.add(Convolution2D(32, 6, 6, border_mode='same', activation='relu', dim_ordering='th', input_shape=_IMAGE_SHAPE))
    model.add(Dropout(0.25))
    model.add(Convolution2D(24, 3, 3, border_mode='same', activation='relu', dim_ordering='th'))
    model.add(Convolution2D(16,  3, 3, border_mode='same', activation='relu', dim_ordering='th'))
    model.add(Convolution2D(8,  3, 3, border_mode='same', activation='relu', dim_ordering='th'))
    model.add(Convolution2D(16,  3, 3, border_mode='same', activation='relu', dim_ordering='th'))

    
    ## Connect and whatnot
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(_CATEGORIES), activation='softmax'))
    
    ## Add optimizer
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    ## compile
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def train_model( Xtrain, Ytrain):
    ## Create Kfold training/testing Generator
    n_splits = 3
    kf = KFold( n_splits=n_splits, shuffle=True )
    model = make_model()

    for train_idx, test_idx in kf.split(Xtrain):
        model.fit(Xtrain[train_idx], Ytrain[train_idx], 
                    batch_size=32, nb_epoch=10, verbose=1, 
                    callbacks=[], validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
    try:
        save_model(model)
    except:
        pass
    return model

def test_model(model,limit=None):
    print("Loading Test dataset...")
    Xtest, Xtest_fn = load_test_data(imshape=_IMAGE_SHAPE, limit=limit)
    print("Preprocessing test image array")
    Xtest = preprocess(Xtest)
    print("Making model predictions on test data")
    predictions = model.predict_classes(Xtest)
    return predictions
    

def save_model(model):
    model_json = model.to_json()
    open('fishes0.json', 'w').write(model_json)
    model.save_weights('fishes0_weights.h5', overwrite=True)

def load_model(model_def_fname, model_weight_fname):
    model = model_from_json(open(model_def_fname).read())
    model.load_weights(model_weight_fname)
    return model

def create_submission(predictions):
    result1 = pd.DataFrame(predictions, columns=_CATEGORIES)
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'ZD_fishey-1_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

    


    

    
    
    


##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    ## Only for linux/mac, simple "press any key to continue" implemntation
    print("Press Any Key to Exit")
    os.system('read')