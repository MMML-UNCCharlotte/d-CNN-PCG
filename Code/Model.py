# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 02:27:35 2020

@author: imrul
"""

import cv2 as cv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K

#############################################################################

def my_RMSE(y_true, y_pred):
    y_true = K.constant(y_true) 
    y_pred = K.constant(y_pred) 
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def model_cnn1(optimizer = "Adam", learn_rate=0.005, n_epoch = 500, preferred_height = 224, preferred_width = 224, n_d=1):
  #Instantiate an empty model
  model = Sequential()
  
  # 1st Convolutional Layer
  model.add(Conv2D(filters=3, input_shape=(preferred_height,preferred_width,n_d), kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # 2nd Convolutional Layer
  model.add(Conv2D(filters=3, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # 3rd Convolutional Layer
  model.add(Conv2D(filters=3, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # 4th Convolutional Layer
  model.add(Conv2D(filters=4, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # Max Pooling
  model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
  
  # 5th Convolutional Layer
  model.add(Conv2D(filters=6, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # Max Pooling
  model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
  
  # 6th Convolutional Layer
  model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # Max Pooling
  model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
  
  # 7th Convolutional Layer
  model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # Max Pooling
  model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
  # 8th Convolutional Layer
  model.add(Conv2D(filters=10, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # Max Pooling
  model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
  # 9th Convolutional Layer
  model.add(Conv2D(filters=10, kernel_size=(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  # Passing it to a Fully Connected layer
  model.add(Flatten())
  
  # Add Dropout
  model.add(Dropout(0.1))

  # 1st Fully Connected Layer
  model.add(Dense(490))
  model.add(Activation('relu'))
  
  # Add Dropout
  model.add(Dropout(0.1))
  
  # Output Layer
  model.add(Dense(1))
  model.add(Activation('linear'))    
  
  return model
