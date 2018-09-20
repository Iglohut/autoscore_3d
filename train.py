#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:23:05 2018

@author: sebastian
"""

from i3d_inception import Inception_Inflated3d
from i3d_generator import i3d_generator
import h5py
from keras.models import Model
from keras.layers import Reshape
from keras.layers import Dense

def train(path_train, batch_size, path_val = None, t_size = 10):
    data = h5py.File(path_train,'r')
        
    X = data['X']
    Y = data['Y']
    input_shape = (t_size,)+X.shape[1:]
    
    rgb_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(input_shape))
    
    output_old = rgb_model.layers[-1].output
    
    x = Reshape((1024,) ,name='Reshape_top')(output_old)
    x = Dense(50,activation = 'selu',name='Dense_top_1')(x)
    x = Dense(1,activation = 'sigmoid',name='Dense_top_2')(x)

    
    model_final = Model(input=rgb_model.input, output=[x])
    model_final.compile(loss = 'mse',optimizer = 'adam')
    
    train_generator = i3d_generator(X, Y, 2, t_size)
    
    
    if path_val:
        
        data_val = h5py.open(path_train,'r')
        X_val = data_val['X']
        Y_val = data_val['Y']
        val_generator = i3d_generator(X_val, Y_val, 5, t_size)
        model_final.fit_generator(train_generator,
                                  steps_per_epoch=2000,
                                  epochs=50,
                                  validation_data=val_generator,
                                  validation_steps=800)
    
    else:
        model_final.fit_generator(train_generator.__getitem__(),
                                  steps_per_epoch=2000,
                                  epochs=50)
        
    
train('/home/sebastian/Desktop/data.h5',10)
    