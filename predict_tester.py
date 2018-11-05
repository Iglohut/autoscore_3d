#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:41:39 2018

@author: deeplabchop
"""

import skvideo.io
import matplotlib.pyplot as plt
import keras
from keras.models import Model, load_model
import numpy as np

input_video = '/home/deeplabchop/trifle/homes/evelien/Calcium imaging/32363-32366/Session 1 10.07-14.07/10.07.2017/32364/Trial1_10072017_2017-07-10-120636-0000.avi'
#"/home/deeplabchop/trifle/homes/evelien/Calcium imaging/32363-32366/Object space/mouse_training_OS_calcium_1_t0001_raw.avi"
model_path = "/home/deeplabchop/src/autoscore_iglo/autoscore_checkpoint"

model_final = load_model(model_path)
print("loaded model")


reader = skvideo.io.vreader(input_video)
vwriter = skvideo.io.FFmpegWriter('/home/deeplabchop/Desktop/vid2_normal.mp4')

b_w = False


frames = []
labels = []
for i,frame in enumerate(reader):
    if i>90:
        
        
        if b_w:
            for f in range (3):
                frame[:,:,f] = frame.mean(-1)
        
        frames.append(frame[:800,100:900,:])
        
        if len(frames) == 9:
    
            X = np.array([frames])
            Y = model_final.predict(X)
            
            current_frame = frames[4][::2,::2,:].copy()#.repeat(2,axis = 0).repeat(2,axis = 1)
            
            if Y[0][0]<0.7:
                color_channel = 0
            else:
                color_channel = 1
                        
            current_frame[-20:,:int(current_frame.shape[1] * Y[0][0]),color_channel]=255 # You are not taking into account to take the middle frame, now you take the last.
            
            print("yey",i, Y)
            
            _ = frames.pop(0)
            
            vwriter.writeFrame(current_frame)
            
            if i>1000:
                break
        
vwriter.close()
