#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:03:23 2018

@author: sebastian
"""

import numpy as np
from keras.utils.data_utils import Sequence
import cv2
import random
import copy


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
    def __init__(self, data, slices,  batch_size, input_frames, n_labels, p_augment = 0.5, p_affine = 0.5, p_resize=0.5, p_noise=0.5):
        self.X = data['X']
        self.Y = data['Y']
        self.slices = np.random.permutation(slices)
        self.batch_size = batch_size
        self.n_frames = input_frames
        self.X_size = (self.n_frames,)+ self.X.shape[1:]
        self.X_size_batch = (self.batch_size, ) + self.X_size
        self.Y_size = n_labels
        self.Y_size_batch = (self.batch_size, 2) # Want to be self.Y_size when more labels

        self.p_augment = p_augment
        self.p_affine = p_affine
        self.p_resize = p_resize
        self.p_noise = p_noise

        self.total_slices = len(self.slices)
        self.cur_slice = 0

    def _mid_label(self, Y):
        return Y[-int(np.ceil(self.n_frames / 2))]

    def _aFineTransform_batch(self):
        def pickrowcol(Xloc, Yloc, rows, cols, discharge=0.04):
            Xpoint = int(random.uniform(Xloc - discharge, Xloc + discharge) * cols)
            Ypoint = int(random.uniform(Yloc - discharge, Yloc + discharge) * rows)
            return [Xpoint, Ypoint]

        batches, frames, rows, cols, ch = self.batch_X.shape

        # Relative object locations
        Ytop = 0.234375
        Ybot = 1 - Ytop
        Xleft = 0.234375
        Xright = 1 - Xleft

        pts_from = []
        pts_to = []

        quads = ['UL', 'UR', 'LL', 'LR']
        for i in range(3): # Pick 3 quadrant point to transform
            quadi = random.choice(quads)
            quads.remove(quadi)
            if quadi == 'UL':
                pts_from.append(pickrowcol(Xloc=Xleft, Yloc=Ytop, rows=rows, cols=cols))
                pts_to.append(pickrowcol(Xloc=Xleft, Yloc=Ytop, rows=rows, cols=cols))

            if quadi == 'UR':
                pts_from.append(pickrowcol(Xloc=Xright, Yloc=Ytop, rows=rows, cols=cols))
                pts_to.append(pickrowcol(Xloc=Xright, Yloc=Ytop, rows=rows, cols=cols))

            if quadi == 'LL':
                pts_from.append(pickrowcol(Xloc=Xleft, Yloc=Ybot, rows=rows, cols=cols))
                pts_to.append(pickrowcol(Xloc=Xleft, Yloc=Ybot, rows=rows, cols=cols))

            if quadi == 'LR':
                pts_from.append(pickrowcol(Xloc=Xright, Yloc=Ybot, rows=rows, cols=cols))
                pts_to.append(pickrowcol(Xloc=Xright, Yloc=Ybot, rows=rows, cols=cols))

        pts_from = np.asarray(pts_from, dtype=np.float32)
        pts_to = np.asarray(pts_to, dtype=np.float32)

        M = cv2.getAffineTransform(pts_from, pts_to)

        for batchi in range(batches):
            for framei in range(frames):
                self.batch_X[batchi, framei] = cv2.warpAffine(self.batch_X[batchi, framei], M, (cols, rows))


    def _resize_batch(self):
        batches, frames, rows, cols, ch = self.batch_X.shape
        factor = random.uniform(0.3, 1.5)

        rows, cols, _ = cv2.resize(self.batch_X[0, 0], None, fx=factor, fy=factor).shape # Would be the new size of image
        batch_X = np.zeros((batches, frames, rows, cols, ch))

        for batchi in range(batches):
            for framei in range(frames):
                batch_X[batchi, framei] = cv2.resize(self.batch_X[batchi, framei], None, fx=factor, fy=factor)

        self.batch_X = batch_X.copy()


    def _noise_batch(self):
        magnitude = random.uniform(-1.5, 1.5) # Noise between -1.5 and 1.5 std
        self.batch_X += magnitude * self.batch_X.std() * np.random.random(self.batch_X.shape)

    def _correct_type(self):
        # Removes values below 0 and rounds pixel values
        super_threshold_indices = self.batch_X < 0
        self.batch_X[super_threshold_indices] = 0
        self.batch_X = np.round(self.batch_X)


    def _augment_batch(self):
        if random.uniform(0, 1) < self.p_augment: # If should augment data at all
            if random.uniform(0,1) < self.p_affine:
                self._aFineTransform_batch() # Affine Transformation on three points

            if random.uniform(0,1) < self.p_resize:
                self._resize_batch() # Resize the input images

            if random.uniform(0,1) < self.p_noise:
                self._noise_batch() # Add random uniform noise

            self._correct_type() # All values to pixel values

    def _get_batch(self):
        self.batch_X = np.zeros(self.X_size_batch)
        self.batch_Y = np.zeros(self.Y_size_batch)

        for slice_i in range(self.batch_size):
            if self.cur_slice >= self.total_slices: # If gone over all slices: shuffle set and start anew
                self.cur_slice = 0
                self.slices = np.random.permutation(self.slices)

            self.batch_X[slice_i] = self.X[self.slices[self.cur_slice]]
            self.batch_Y[slice_i, 0] = self._mid_label(self.Y[self.slices[self.cur_slice]])
            self.batch_Y[slice_i, 1] = int(not self.batch_Y[slice_i, 0])

            self.cur_slice += 1


    def __call__(self):
        self._get_batch() # Preprare general batches
        self._augment_batch() # Do data augmentation

        return self.batch_X, self.batch_Y