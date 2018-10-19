#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:23:05 2018

@author: sebastian
"""

from i3d_inception import Inception_Inflated3d
from i3d_generator import i3d_generator
import h5py
from keras.models import Model, load_model
from keras.layers import Reshape
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import os

def train(path_train, batch_size, path_val = None, t_size = 9,train_val_split=1,validate=True):
    data = h5py.File(path_train,'r')
        
    X = data['X']
    Y = data['Y']
    input_shape = (t_size,)+X.shape[1:]
    
    if 'autoscore_checkpoint' in os.listdir('.'):
        model_final = load_model('autoscore_checkpoint')
        print('Loaded existing model')
        
    else:
        
        rgb_model = Inception_Inflated3d(
                    include_top=False,
                    weights='rgb_imagenet_and_kinetics',
                    input_shape=(input_shape))
        
        output_old = rgb_model.layers[-1].output
        
        x = Reshape((1024,) ,name='Reshape_top')(output_old)
        x = Dense(50,activation = 'selu',name='Dense_top_1')(x)
        x = Dense(2,activation = 'sigmoid',name='Dense_top_2')(x)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_final = Model(input=rgb_model.input, output=[x])
        model_final.compile(loss = 'binary_crossentropy',optimizer = sgd,
                  metrics=['mae', 'acc'])
    checkpoint = ModelCheckpoint('autoscore_checkpoint', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model_final = load_model('autoscore_checkpoint')
    train_generator = i3d_generator(X, Y, 7, t_size,train_val_split=0.9,train=True )
    
    if validate:
        val_generator = i3d_generator(X, Y, 7, t_size,train_val_split=0.9,train=False )
        model_final.fit_generator(train_generator.__getitem__(),
                              steps_per_epoch=5000,
                              epochs=50,
                              validation_data=val_generator.__getitem__(),
                              validation_steps=1000,
                              callbacks = [checkpoint])
    
#    elif path_val:
#        
#        data_val = h5py.open(path_train,'r')
#        X_val = data_val['X']
#        Y_val = data_val['Y']
#        val_generator = i3d_generator(X_val, Y_val, 5, t_size)
#        model_final.fit_generator(train_generator.__getitem__(),
#                                  steps_per_epoch=2000,
#                                  epochs=50,
#                                  validation_data=val_generator.__getitem__(),
#                                  validation_steps=800)
#    
#    else:
#        if validate:
#            model_final.fit_generator(train_generator.__getitem__(),
#                                      steps_per_epoch=2000,
#                                      epochs=50)
    model_final.save('autoscore_model_3')        
    
train('/home/deeplabchop/src/autoscore_3d/data/data.h5',10)






    