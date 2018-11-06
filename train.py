#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:23:05 2018

@author: sebastian
"""

from i3d_generator import i3d_generator
import h5py
from keras.callbacks import ModelCheckpoint
import os
from pathlib import Path
from OS_utils import read_yaml
from network import get_network


def train(config):

    # Setting parameters
    batch_size = config['model_params']['batch_size']
    t_size = config['model_params']['n_frames']
    val_split = config['model_params']['train_val_split']
    n_epochs = config['model_params']['n_epochs']
    n_iters_train = config['model_params']['n_iters_train']
    n_iters_val = config['model_params']['n_iters_val']
    model_name = config['model_params']['name']


    # Load data
    data = h5py.File(config['dataset'], 'r')
    X = data['X']
    Y = data['Y']

    model_final = get_network(model_name) # Creating or loading model

    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    train_generator = i3d_generator(X, Y, batch_size, t_size,train_val_split=val_split,train=True )

    val_generator = i3d_generator(X, Y, batch_size, t_size,train_val_split=val_split,train=False )


    

    model_final.fit_generator(train_generator.__getitem__(),
                          steps_per_epoch= n_iters_train,
                          epochs= n_epochs,
                          validation_data=val_generator.__getitem__(),
                          validation_steps= n_iters_val,
                          callbacks = [checkpoint])
    

    model_final.save(model_name)





# Use function:
config = read_yaml(Path(os.getcwd()).resolve() / 'config.yaml')
train(config)




# train('/home/sebastian/Desktop/data.h5',10)

