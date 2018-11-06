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
from keras.callbacks import ModelCheckpoint
import os

from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.layers import AveragePooling3D
from pathlib import Path
from OS_utils import read_yaml



def get_network(model_name):
    input_shape = (None, None, None, 3)

    if model_name in os.listdir('.'):
        model_final = load_model('autoscore_checkpoint')
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

    return model_final