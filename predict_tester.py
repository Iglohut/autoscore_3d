#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:41:39 2018

@author: deeplabchop
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import skvideo.io
import matplotlib.pyplot as plt
import keras
from keras.models import Model, load_model
import numpy as np
import h5py
from skimage.exposure import cumulative_distribution
from OS_utils import read_yaml
from pathlib import Path
import cv2
from PredictUtils.utils import cdf, hist_matching, im_equalize_hist

# def cdf(im):
#      '''
#      computes the CDF of an image im as 2D numpy ndarray
#      '''
#      c, b = cumulative_distribution(im)
#      # pad the beginning and ending pixels and their CDF values
#      c = np.insert(c, 0, [0]*b[0])
#      c = np.append(c, [1]*(255-b[-1]))
#      return c
#
# def hist_matching(c, c_t, im):
#      '''
#      c: CDF of input image computed with the function cdf()
#      c_t: CDF of template image computed with the function cdf()
#      im: input image as 2D numpy ndarray
#      returns the modified pixel values
#      '''
#      pixels = np.arange(256)
#      # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of
#      # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
#      new_pixels = np.interp(c, c_t, pixels)
#      im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
#      return im
#
#
# def im_equalize_hist(baseframe, frame):
#     newim = np.zeros(frame.shape, dtype=np.int)
#     for i in range(frame.shape[-1]):
#         c_t = cdf(baseframe[:,:,i].astype(int))
#         c = cdf(frame[:,:,i].astype(int))
#         newim[:,:,i] = hist_matching(c, c_t, frame[:,:,i])
#     return newim





# Setting parameters
n_frames = 9
n_frames_steps = 3


# Some paths
#input_video = '/home/deeplabchop/trifle/homes/evelien/Calcium imaging/32363-32366/Object space/mouse_training_OS_calcium_1_t0001_raw.avi'
#input_video = '/home/deeplabchop/src/autoscore_iglo2/mouse_training_OS_5trials_inteldis_59_66_or_206_13_t0001_raw.avi'
input_video = "/home/deeplabchop/src/autoscore_iglo2/mouse_training_OS_5trials_inteldis_36_44_or_101_9_t0003_raw.avi"
#input_video = "/home/deeplabchop/src/autoscore_iglo2/rat_training_OS_baseline_ repeat_2_t0001_raw.avi"
model_path = "/home/deeplabchop/src/autoscore_iglo2/project/henk_bigger_3frames_checkpoint"
#model_path = "/home/deeplabchop/src/autoscore_3d/autoscore_model_2"
data_path ='/home/deeplabchop/src/autoscore_3d/data/data.h5'
model_final = load_model(model_path)
print("loaded model")


reader = skvideo.io.vreader(input_video)
vwriter = skvideo.io.FFmpegWriter('/home/deeplabchop/Desktop/vid_henkbig3_greenOS.mp4')

# Extra stuff
data = h5py.File(data_path, 'r')
baseframe = data["X"][::10000,:,:].mean(0)




# Preprocessing stuff

input_shape = tuple(np.shape(model_final.input)[2:4]) # Height and Width
#cv2.resize(frame, input_shape)
video_shape = skvideo.io.FFmpegReader(input_video).getShape()
total_frames = video_shape[0]



frames = []
labels = []
for i,frame in enumerate(reader):
    
    # Some preprocessing
    frame = im_equalize_hist(baseframe, frame)
#    frame = cv2.resize(frame, (512, 384))
    frame = cv2.resize(frame.astype("uint8"), (512, 384))
#    if input_shape[0] and (not video_shape[1:3] == input_shape): cv2.resize(frame, input_shape)
    
    
    
    
    if i>400 and i <total_frames-400: # Predicter loop
        
        frames.append(frame)
        
        if len(frames) == n_frames * n_frames_steps: 
    
            X = np.array([frames[slice(0, n_frames * n_frames_steps, n_frames_steps)]])
            Y = model_final.predict(X)
            
            current_frame = frames[4][::2,::2,:].copy()#.repeat(2,axis = 0).repeat(2,axis = 1)
            
            if Y[0][0]<0.5: # Colorbar
                color_channel = 0
            else:
                color_channel = 1
                        
            current_frame[-20:,:int(current_frame.shape[1] * Y[0][0]),color_channel]=255  # Size of colorbar
            
            print("yey",i, Y)
            
            _ = frames.pop(0)
            
            vwriter.writeFrame(current_frame)
            
#            if i>1000:
#                break
        
vwriter.close()
