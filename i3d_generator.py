#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:03:23 2018

@author: sebastian
"""

import numpy as np
from keras.utils.data_utils import Sequence

class i3d_generator(Sequence):
    def __init__(self, X, Y, batch_size, input_frames, randomize = True):
        self.X = X
        self.Y = Y 
        input_size = (input_frames,)+X.shape[1:]
        self.input_x = np.zeros((batch_size,)+input_size)
        self.input_y = np.zeros((batch_size,)+(1,))        
        self.batch_size = batch_size
        self.input_size = input_size
        self.idcs = np.arange(X.shape[0]-input_size[-1]-1)
        if randomize:
            self.idcs = np.random.permutation(self.idcs)
        self.idx = 0  
    def __len__(self):
        return(self.batch_size)
    def __getitem__(self, batch = None):
        while 1:                        
            for batch in range(self.batch_size):
                self.input_x[batch,:,:,:]  = self.X[self.idcs[self.idx]:self.idcs[self.idx]+self.input_size[0],:,:]
                self.input_y[batch,:]      = self.Y[self.idcs[self.idx]+(self.input_size[0]//2)] 
                self.idx = (self.idx+1)%(len(self.idcs))
            yield(self.input_x.copy(),self.input_y.copy())
