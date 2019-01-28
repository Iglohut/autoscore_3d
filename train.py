import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

from i3d_generator import i3d_generator, SS_generator
import h5py
import os
from pathlib import Path
from OS_utils import read_yaml, get_slices, Logger
from network import get_network, noob_network, get_network_bigger, original_networkish, ST_network


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
    n_frames_steps = config['model_params']['n_frames_steps']
    val_split = config['model_params']['train_val_split']
    n_epochs = config['model_params']['n_epochs']
    n_iters_train = config['model_params']['n_iters_train']
    n_iters_val = config['model_params']['n_iters_val']
    model_name = config['model_params']['name']
    project_path = os.getcwd() + '/project/'
    model_path = project_path + model_name + '_checkpoint'

    # Load data
    data = h5py.File(config['dataset'], 'r')


    # Creating or loading model
#    model_final = get_network(model_path)
    model_final = get_network_bigger(model_path, opt='RMSprop')
#     model_final = noob_network() # Weak model with same input-output to debug
#    model_final = original_networkish(model_path, (n_frames,) + data["X"].shape[1:])
#     model_final = ST_network(model_path, (n_frames,) + data["X"].shape[1:])
    

    # Metric logger
    logger = Logger(project_path, model_name)

    # Training vs validation generator
    slices_train, slices_val = get_slices(data, project_path, model_name, n_frames=n_frames, val_split=val_split, n_stacker = 1, steps = n_frames_steps)

    generator_train = SS_generator(data = data, slices = slices_train,  batch_size = batch_size, input_frames = n_frames, n_labels = n_behaviours, p_resize= -1, p_augment = 0.8)
    generator_val = SS_generator(data = data, slices = slices_val, batch_size = batch_size, input_frames = n_frames, n_labels = n_behaviours, p_augment = -1)


    for epochi in range(logger.start_epoch, n_epochs + 1):
        # Metrics for AUROC
        Y_reals =[]
        Y_preds =[]
        metrics_train =[]

        for traini in range(n_iters_train): # Training loop
            X, Y = generator_train()
            metrics_train.append(model_final.train_on_batch(X, Y)) # This trains the model!


            Y_predicted = model_final.predict_on_batch(X)
            Y_reals = np.append(copy.copy(Y_reals), copy.copy(Y[:,0]))
            Y_preds = np.append(copy.copy(Y_preds), copy.copy(Y_predicted[:,0]))

        fpr_train, tpr_train, thresholds_train = roc_curve(Y_reals, Y_preds)
        auc_train = auc(fpr_train, tpr_train)
        metrics_train_mean = np.mean(metrics_train, axis=0)


        # Metrics for AUROC
        Y_reals =[]
        Y_preds =[]
        metrics_val=[]
        for vali in range(n_iters_val): # Validation loop
            X, Y = generator_val()
            metrics_val.append(model_final.test_on_batch(X, Y))

            Y_predicted = model_final.predict_on_batch(X)
            Y_reals = np.append(copy.copy(Y_reals), copy.copy(Y[:, 0]))
            Y_preds = np.append(copy.copy(Y_preds), copy.copy(Y_predicted[:, 0]))

        fpr_val, tpr_val, thresholds_val = roc_curve(Y_reals, Y_preds)
        auc_val = auc(fpr_val, tpr_val)
        metrics_val_mean = np.mean(metrics_val, axis=0)


        print("Epoch {}: train loss: {:.5f}, train mae: {:.5f}, train acc: {:.3f}, train auc: {:.3f} | val loss: {:.5f}, val mae: {:.5f}, val acc: {:.3f}, val auc: {:.3f}".format(
                                                                               str(epochi) +'/'+str(n_epochs),
                                                                               metrics_train_mean[0],
                                                                               metrics_train_mean[1],
                                                                               metrics_train_mean[2],
                                                                               auc_train,
                                                                               metrics_val_mean[0],
                                                                               metrics_val_mean[1],
                                                                               metrics_val_mean[2],
                                                                                   auc_val))
        # Saving stuff
        model_final.save(model_path)
        logger.store(epochi, metrics_train_mean[0],
        metrics_train_mean[1],
        metrics_train_mean[2],
        auc_train,
        metrics_val_mean[0],
        metrics_val_mean[1],
        metrics_val_mean[2],
        auc_val)



    # Use function:
train()




# import matplotlib.pyplot as plt
# import cv2
# import random
#
#
#
# X, Y = generator_train()
#
# frame = X[0,0,:]
#
#
# plt.imshow(frame.astype(np.int32)),plt.title('Input'), plt.axis('off')
#
# #
# # plt.subplot(121),plt.imshow(frame.astype(np.int32)),plt.title('Input'), plt.axis('off')
# # plt.subplot(122),plt.imshow(noised_image.astype(np.int32)),plt.title('Output'), plt.axis('off')
