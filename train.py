from i3d_inception import Inception_Inflated3d
from i3d_generator import i3d_generator
import h5py
from keras.models import Model
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.layers import AveragePooling3D


def train(path_train, batch_size, path_val = None, t_size = 10):
    data = h5py.File(path_train,'r')
        
    X = data['X']
    Y = data['Y']
    input_shape = (t_size,)+X.shape[1:]

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
        model_final.fit_generator(train_generator.__getitem__(),
                                  steps_per_epoch=2000,
                                  epochs=50)
        
    
train('/home/sebastian/Desktop/data.h5',10)