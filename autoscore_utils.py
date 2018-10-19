#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:36:53 2018

@author: sebastian
"""

import pandas as pd
import os
import h5py
import skvideo.io
import numpy as np 
import matplotlib.pyplot as plt

n_trial = 0
current_frame=0
n_trial_category = {}

vid_root_1 = '/home/deeplabchop/trifle/homes/evelien/Calcium imaging/32363-32366/Object space'
vid_root_2 = '/home/deeplabchop/trifle/homes/evelien/Calcium imaging/32363-32366/Object space/11.07.2017-12.07.2017 os videos'

vids = sorted([os.path.join(vid_root_1,f) for f in os.listdir(vid_root_1) if '.avi'  in f and 'raw' in f])[:20]
vids1 = sorted([os.path.join(vid_root_2,f)  for f in os.listdir(vid_root_2) if '.avi'  in f and  'raw' in f])
vids2 = sorted([os.path.join(vid_root_1,f) for f in os.listdir(vid_root_1) if '.avi'  in f and  'raw' in f])[20:]
vids_raw = vids+vids1+vids2 

vids = sorted([os.path.join(vid_root_1,f) for f in os.listdir(vid_root_1) if '.avi'  in f and not 'raw' in f])[:20]
vids1 = sorted([os.path.join(vid_root_2,f)  for f in os.listdir(vid_root_2) if '.avi'  in f and not 'raw' in f])
vids2 = sorted([os.path.join(vid_root_1,f) for f in os.listdir(vid_root_1) if '.avi'  in f and not 'raw' in f])[20:]
vids = vids+vids1+vids2 

for vid in vids:
    print(vid)
    
f = h5py.File('data/data.h5','w')

X = f.create_dataset('X',shape = (1, 384, 512, 3),maxshape = (None, 384, 512, 3),chunks=(7, 384, 512, 3),compression='gzip')
Y = f.create_dataset('Y',shape = (1,), maxshape = (None,))

position = 0
delimiters = [position]

for i, vid in enumerate(vids):
    
    print(str(i)+'/'+str(len(vids)))
    
    videodata = skvideo.io.vread(vid) 
    
    blackness = np.diff(((videodata[:,0:5,0:5,:].sum(3)==0).sum(1).sum(1)>0).astype(np.int))
    start = np.argmax(blackness)
    end = np.where(blackness==blackness.min())[-1][-1]
    
    videodata = videodata[start:end]
    
    redness = (np.logical_and(videodata[:,:,:,0]==255,videodata[:,:,:,1]==0,videodata[:,:,:,2]==0)).sum(1).sum(1)>1000
    
    videodata = skvideo.io.vread(vids_raw[i])[start:end,:,:,:]
    
    position
    
    X.resize([position+videodata.shape[0], 384, 512, 3])
    Y.resize([position+videodata.shape[0]])
    X[position:position+videodata.shape[0],:,:,:] = videodata
    Y[position:position+videodata.shape[0]]=redness
    position+=videodata.shape[0]
    
    delimiters.append(position) 
    

f['delimiters']=delimiters

testvid = np.zeros((1000,384, 512, 3))

start = 1000
idcs = f['Y'][start:start+1000]
testvid = f['X'][start:start+1000,:,:,:]
testvid[idcs.astype(np.bool),0:10,0:10,:]=255

skvideo.io.vwrite('data/test.avi',testvid)
f.close()

