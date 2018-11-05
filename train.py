<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:23:05 2018

@author: sebastian
"""
from keras import backend as K
=======
>>>>>>> 956f8b3e157f8f15ffc5dd9b6457fa218dc4b170
from i3d_inception import Inception_Inflated3d
from i3d_generator import i3d_generator
import h5py
from keras.models import Model, load_model
from keras.layers import Reshape
from keras.layers import Dense
<<<<<<< HEAD
from keras.layers import Conv3D, Conv1D, Conv2D, Lambda
from keras.layers import BatchNormalization
from keras.layers import AveragePooling3D
from keras.layers import Flatten, Average, Activation
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import os
=======
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.layers import AveragePooling3D

>>>>>>> 956f8b3e157f8f15ffc5dd9b6457fa218dc4b170

def train(path_train, batch_size, path_val = None, t_size = 9,train_val_split=1,validate=True):
    data = h5py.File(path_train,'r')
        
    X = data['X']
    Y = data['Y']
    input_shape = (t_size,)+X.shape[1:]
<<<<<<< HEAD
    input_shape = (None, None, None, 3)
    
    if 'autoscore_checkpoint' in os.listdir('.'):
        model_final = load_model('autoscore_checkpoint')
        print('Loaded existing model')
        
    else:
        print("Creating new model, for you!")
        rgb_model = Inception_Inflated3d(
                    include_top=False,
                    weights='rgb_imagenet_and_kinetics',
                    input_shape=(input_shape))
        
#        output_old = rgb_model.layers[-1].output
#        
#        x = Reshape((1024,) ,name='Reshape_top')(output_old)
#        x = Dense(50,activation = 'selu',name='Dense_top_1')(x)
#        x = Dense(2,activation = 'sigmoid',name='Dense_top_2')(x)
        
            # Refining the network
        rgb_model.layers.pop()  # Deleting the last AveragePooling3D Layer
        output_old = rgb_model.layers[-1].output
    
        x = Conv3D(2, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
                   dilation_rate=(1, 1, 1), activation=None)(output_old)
#        x = BatchNormalization()(x)
#        x = Conv3D(256, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
#                   dilation_rate=(1, 1, 1), activation='relu')(x)
#        x = BatchNormalization()(x)
#        x = Conv3D(128, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
#                   dilation_rate=(1, 1, 1), activation='relu')(x)
#        x = BatchNormalization()(x)
#        x = Conv3D(64, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
#                   dilation_rate=(1, 1, 1), activation='relu')(x)
#        x = BatchNormalization()(x)
#        x = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
#                   dilation_rate=(1, 1, 1), activation='relu')(x)
#        x = BatchNormalization()(x)
#        x = Conv3D(2, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
#                   dilation_rate=(1, 1, 1), activation='relu')(x) # End with one filter because for now filter should represent probability of 1 behaviour type; dense on more filters make more sense?
#        x = AveragePooling3D(pool_size=(2, 14, 14))(x)
        x = Lambda(lambda x: K.mean(x, axis = -2))(x)
        x = Lambda(lambda x: K.mean(x, axis = -2))(x)
        x = Lambda(lambda x: K.mean(x, axis = -2))(x)
        x = Activation('softmax')(x)
#        x = Lambda(lambda x: x)(x)
#        x = Reshape((2,), name='Reshape_top')(x)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_final = Model(input=rgb_model.input, output=[x])
        model_final.compile(loss = 'binary_crossentropy',optimizer = sgd,
                  metrics=['mae', 'acc'])
    checkpoint = ModelCheckpoint('autoscore_checkpoint', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    train_generator = i3d_generator(X, Y, 7, t_size,train_val_split=0.9,train=True )
    
    if validate:
        val_generator = i3d_generator(X, Y, 7, t_size,train_val_split=0.9,train=False )
=======

    rgb_model = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(input_shape))


    # Refining the network
    rgb_model.layers.pop()  # Deleting the last AveragePooling3D Layer
    output_old = rgb_model.layers[-1].output

    x = Conv3D(512, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
               dilation_rate=(1, 1, 1), activation=None)(output_old)
    x = BatchNormalization()(x)
    x = Conv3D(256, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
               dilation_rate=(1, 1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(128, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
               dilation_rate=(1, 1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
               dilation_rate=(1, 1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(16, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
               dilation_rate=(1, 1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv3D(1, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
               dilation_rate=(1, 1, 1), activation='relu')(x) # End with one filter because for now filter should represent probability of 1 behaviour type; dense on more filters make more sense?
    x = AveragePooling3D(pool_size=(2, 4, 4))(x)
    x = Reshape((1,), name='Reshape_top')(x)

    model_final = Model(input=rgb_model.input, output=[x])
    model_final.compile(loss='mse', optimizer='adam')

    
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
>>>>>>> 956f8b3e157f8f15ffc5dd9b6457fa218dc4b170
        model_final.fit_generator(train_generator.__getitem__(),
                              steps_per_epoch=1000,
                              epochs=1000,
                              validation_data=val_generator.__getitem__(),
                              validation_steps=1000,
                              callbacks = [checkpoint])
    
<<<<<<< HEAD
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






    
=======
train('/home/sebastian/Desktop/data.h5',10)
>>>>>>> 956f8b3e157f8f15ffc5dd9b6457fa218dc4b170
