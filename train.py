from i3d_generator import i3d_generator, SS_generator
import h5py
from keras.callbacks import ModelCheckpoint
import os
from pathlib import Path
from OS_utils import read_yaml, get_slices
from network import get_network, noob_network


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import copy
import numpy as np

def train():
    # Get config file
    config = read_yaml(Path(os.getcwd()).resolve() / 'config.yaml')

    # Setting parameters
    n_behaviours = len(config['behaviours'])
    batch_size = config['model_params']['batch_size']
    n_frames = config['model_params']['n_frames']
    val_split = config['model_params']['train_val_split']
    n_epochs = config['model_params']['n_epochs']
    n_iters_train = config['model_params']['n_iters_train']
    n_iters_val = config['model_params']['n_iters_val']
    model_name = config['model_params']['name']
    project_path = os.getcwd() + '/project/'
    model_path = project_path + model_name

    # Load data
    data = h5py.File(config['dataset'], 'r')
    # X = data['X']
    # Y = data['Y']

    # Creating or loading model
    # model_final = get_network(model_path)
    model_final = noob_network() # Weak model with same input-output to test

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


    # Training vs validation generator
    slices_train, slices_val = get_slices(data, project_path, n_frames=n_frames, val_split=val_split)

    # train_generator = i3d_generator(X, Y, batch_size, n_frames,train_val_split=val_split,train=True )
    #
    # val_generator = i3d_generator(X, Y, batch_size, n_frames,train_val_split=val_split,train=False )


    generator_train = SS_generator(data = data, slices = slices_train,  batch_size = batch_size, input_frames = n_frames, n_labels = n_behaviours)
    generator_val = SS_generator(data=data, slices=slices_val, batch_size=batch_size, input_frames=n_frames, n_labels=n_behaviours)



    # Per epoch
    Y_reals =[]
    Y_preds =[]
    metrics_train =[]
    metric_names = model_final.metrics_names

    for i in range(n_iters_train):
    # Per iteration
        X, Y = generator_train()
        metrics_train.append(model_final.train_on_batch(X, Y))


        Y_predicted = model_final.predict_on_batch(X)
        Y_reals = np.append(copy.copy(Y_reals), copy.copy(Y[:,0]))
        Y_preds = np.append(copy.copy(Y_preds), copy.copy(Y_predicted[:,0]))

        fpr_train, tpr_train, thresholds_train = roc_curve(Y_reals, Y_preds)
        auc_train = auc(fpr_train, tpr_train)



        metrics_train_mean = np.mean(metrics_train, axis=0)
        print("train loss: {}, train mae: {}, train acc: {}, train auc: {}".format(metrics_train_mean[0], metrics_train_mean[1], metrics_train_mean[2], auc_train), end='\r')


    # model_final.test_on_batch(X, Y)


    # model_final.fit_generator(train_generator.__getitem__(),
    #                       steps_per_epoch= n_iters_train,
    #                       epochs= n_epochs,
    #                       validation_data=val_generator.__getitem__(),
    #                       validation_steps= n_iters_val,
    #                       callbacks = [checkpoint])
    #
    #
    # model_final.save(model_path)





# Use function:
train()







# TODO Make this a slices_train, slices_val function to input SS_generator(data, slices) with __call__
# TODO Try to equal the labels out somehow (from slices_x?): if this works, maybe create overlapping slices instead of stack





# data['Y'][slices_val[0]][[-int(np.ceil(t_size/2))]][0]

# train('/home/sebastian/Desktop/data.h5',10)





