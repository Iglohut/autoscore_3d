from i3d_generator import i3d_generator, SS_generator
import h5py
import os
from pathlib import Path
from OS_utils import read_yaml, get_slices, Logger
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
    model_path = project_path + model_name + '_checkpoint'

    # Load data
    data = h5py.File(config['dataset'], 'r')


    # Creating or loading model
    # model_final = get_network(model_path)
    model_final = noob_network() # Weak model with same input-output to debug


    # Metric logger
    logger = Logger(project_path, model_name)

    # Training vs validation generator
    slices_train, slices_val = get_slices(data, project_path, model_name, n_frames=n_frames, val_split=val_split)

    generator_train = SS_generator(data = data, slices = slices_train,  batch_size = batch_size, input_frames = n_frames, n_labels = n_behaviours)
    generator_val = SS_generator(data = data, slices = slices_val, batch_size = batch_size, input_frames = n_frames, n_labels = n_behaviours)


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


# data['Y'][slices_val[0]][[-int(np.ceil(t_size/2))]][0]

# train('/home/sebastian/Desktop/data.h5',10)





