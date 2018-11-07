#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:03:23 2018

@author: sebastian
"""

import numpy as np
from keras.utils.data_utils import Sequence

class i3d_generator(Sequence):
    def __init__(self, X, Y, batch_size, input_frames, randomize = True, train_val_split = 1,train = True ):
        self.X = X
        self.Y = Y 
        input_size = (input_frames,)+X.shape[1:]
        self.input_x = np.zeros((batch_size,)+input_size)
        self.input_y = np.zeros((batch_size,)+(2,))        
        self.batch_size = batch_size
        self.input_size = input_size
        if train:
            self.idcs = np.arange(0, int((X.shape[0]-input_size[0]-1)*train_val_split), 3)
        else:
            self.idcs = np.arange(int((X.shape[0]-input_size[0]-1)*train_val_split),X.shape[0]-input_size[0]-1,3)
            
        if randomize:
            self.idcs = np.random.permutation(self.idcs)
        self.idx = 0  
    def __len__(self):
        return(self.batch_size)
    def __getitem__(self, batch = None):
        while 1:                        
            for batch in range(self.batch_size):
                self.input_x[batch,:,:,:]  = self.X[self.idcs[self.idx]:self.idcs[self.idx]+self.input_size[0],:,:]
                self.input_y[batch,0]      = self.Y[self.idcs[self.idx]+(self.input_size[0]//2)]
                self.input_y[batch,1]      = 1-self.Y[self.idcs[self.idx]+(self.input_size[0]//2)]
                self.idx = (self.idx+1)%(len(self.idcs))
            yield(self.input_x.copy(),self.input_y.copy())




class SS_generator(Sequence):
    def __init__(self, data, slices,  batch_size, input_frames, n_labels):
        self.X = data['X']
        self.Y = data['Y']
        self.slices = slices
        self.batch_size = batch_size
        self.n_frames = input_frames
        self.X_size = (self.n_frames,)+ self.X.shape[1:]
        self.X_size_batch = (self.batch_size, ) + self.X_size
        self.Y_size = n_labels
        self.Y_size_batch = (self.batch_size, 2) # Want to be self.Y_size when more labels

        self.total_slices = len(self.slices)
        self.cur_slice = 0

    def _mid_label(self, Y):
        return Y[-int(np.ceil(self.n_frames / 2))]

    def __call__(self):
        self.batch_X = np.zeros(self.X_size_batch)
        self.batch_Y = np.zeros(self.Y_size_batch)

        for slice_i in range(self.batch_size):
            if self.cur_slice > self.total_slices: # If gone over all slices: shuffle set and start anew
                self.cur_slice = 0
                self.slices = np.random.permutation(self.slices)

            self.batch_X[slice_i] = self.X[self.slices[self.cur_slice]]
            self.batch_Y[slice_i, 0] = self._mid_label(self.Y[self.slices[self.cur_slice]])
            self.batch_Y[slice_i, 1] = int(not self.batch_Y[slice_i, 0])

            self.cur_slice += 1
        return self.batch_X, self.batch_Y