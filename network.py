from keras import backend as K

from i3d_inception import Inception_Inflated3d
from i3d_generator import i3d_generator
import h5py
from keras.models import Model, load_model
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Conv3D, Conv1D, Conv2D, Lambda
from keras.layers import BatchNormalization
from keras.layers import AveragePooling3D
from keras.layers import Flatten, Average, Activation
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os

from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.layers import AveragePooling3D
from pathlib import Path
from OS_utils import read_yaml

from i3d_inception import conv3d_bn

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


def get_network_bigger(model_path, opt='SGD'):
    input_shape = (None, None, None, 3)

    if Path(model_path).exists():
        model_final = load_model(model_path)
#        K.set_value(model_final.optimizer.lr, 0.001)
#        K.set_value(model_final.optimizer.momentum, 0.7)
        print('Loaded existing model')

    else:
        print("Creating new model for you!")
        rgb_model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_shape=(input_shape))


        # Refining the network
        rgb_model.layers.pop()  # Deleting the last AveragePooling3D Layer
        output_old = rgb_model.layers[-1].output
#
#        x = Conv3D(2, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
#                   dilation_rate=(1, 1, 1), activation=None)(output_old)

        x = conv3d_bn(output_old, 5, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv_last1')
        x = conv3d_bn(x, 2, 3, 3, 3, strides=(1, 1, 1),use_activation_fn=False, padding='same', name='Conv_last2')

        x = Lambda(lambda x: K.mean(x, axis=-2))(x)
        x = Lambda(lambda x: K.mean(x, axis=-2))(x)
        x = Lambda(lambda x: K.mean(x, axis=-2))(x)
        x = Activation('softmax')(x)
        #        x = Lambda(lambda x: x)(x)
        #        x = Reshape((2,), name='Reshape_top')(x)

        model_final = Model(input=rgb_model.input, output=[x])
       
    if opt == 'SGD':
        opt_model = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    elif opt == 'RMSprop':
        opt_model = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        
    model_final.compile(loss='binary_crossentropy', optimizer=opt_model,
                        metrics=['mae', 'acc'])

    model_final.summary()
    return model_final




def get_network(model_path):
    input_shape = (None, None, None, 3)

    if Path(model_path).exists():
        model_final = load_model(model_path)
        print('Loaded existing model')

    else:
        print("Creating new model for you!")
        rgb_model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_shape=(input_shape))


        # Refining the network
        rgb_model.layers.pop()  # Deleting the last AveragePooling3D Layer
        output_old = rgb_model.layers[-1].output

        x = Conv3D(2, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
                   dilation_rate=(1, 1, 1), activation=None)(output_old)

        x = Lambda(lambda x: K.mean(x, axis=-2))(x)
        x = Lambda(lambda x: K.mean(x, axis=-2))(x)
        x = Lambda(lambda x: K.mean(x, axis=-2))(x)
        x = Activation('softmax')(x)
        #        x = Lambda(lambda x: x)(x)
        #        x = Reshape((2,), name='Reshape_top')(x)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_final = Model(input=rgb_model.input, output=[x])
        model_final.compile(loss='binary_crossentropy', optimizer=sgd,
                            metrics=['mae', 'acc'])

    model_final.summary()
    return model_final




def original_networkish(model_path, input_shape):
    # n_frames = 19
    # input_shape = (n_frames, 384, 512, 3) # Generalize later
    # input_shape = (None, None, None, 3)
    
    if Path(model_path).exists():
        model_final = load_model(model_path)
        print('Loaded existing model')

    else:
        print("Creating new model for you!")
        rgb_model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_shape=(input_shape))

        
        output_old = rgb_model.layers[-1].output

        h = int(output_old.shape[2])
        w = int(output_old.shape[3])

        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(output_old)
        # x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(output_old)

        x = conv3d_bn(x, 2, 1, 1, 1, padding='same',
                      use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, 2))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)
        x = Activation('softmax', name='prediction')(x)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_final = Model(input=rgb_model.input, output=[x])
        model_final.compile(loss = 'binary_crossentropy',optimizer = sgd,
                  metrics=['mae', 'acc'])
    model_final.summary()
    return model_final



def ST_network(model_path, input_shape):
    if Path(model_path).exists():
        model_final = load_model(model_path)
        print('Loaded existing model')

    else:
        print("Creating new model for you!")
        rgb_model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_imagenet_and_kinetics',
            input_shape=(input_shape))

        output_old = rgb_model.layers[-1].output

        h = int(output_old.shape[2])
        w = int(output_old.shape[3])

        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(output_old)

        x = Reshape((1024,) ,name='Reshape_top')(x)
        x = Dense(50,activation = 'selu',name='Dense_top_1')(x)
        x = Dense(2,activation = 'sigmoid',name='Dense_top_2')(x)
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_final = Model(input=rgb_model.input, output=[x])
        model_final.compile(loss='binary_crossentropy', optimizer=sgd,
                            metrics=['mae', 'acc'])
    model_final.summary()
    return model_final


def noob_network():
    print("Making noob network!")
    from keras.models import Sequential
    model = Sequential()
    input_shape = (None, None, None, 3)
    model.add(Conv3D(2, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid', data_format=None,
                       dilation_rate=(1, 1, 1), activation=None, input_shape=input_shape))
    model.add(Lambda(lambda x: K.mean(x, axis=-2)))
    model.add(Lambda(lambda x: K.mean(x, axis=-2)))
    model.add(Lambda(lambda x: K.mean(x, axis=-2)))
    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                        metrics=['mae', 'acc'])

    model.summary()
    return model


