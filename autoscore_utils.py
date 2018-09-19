#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:36:53 2018

@author: sebastian
"""

import pandas as pd
import os
import h5py

summary = pd.read_csv('/home/sebastian/code/keras-kinetics-i3d/data/csv/mouse_training_OS_calcium_1.csv')

data_1    = pd.read_csv('/home/sebastian/code/keras-kinetics-i3d/data/csv/mouse_training_OS_calcium_1_log.csv')
data_2    = pd.read_csv('/home/sebastian/code/keras-kinetics-i3d/data/csv/mouse_training_OS_calcium_log 2.csv')

data = pd.concat([data_1,data_2])
final_sheet = {'subject':[],'n_trial':[],'frame':[],'start_stop':[]}

n_trial = 0
current_frame=0
n_trial_category = {}
for i,row in data.iterrows():
    
    if row.type == 'TR' :
        if row.start_stop==False:
            n_trial+=1
            
        else:
            if summary.iloc[n_trial].subject in n_trial_category.keys():
                n_trial_category[summary.iloc[n_trial].subject]+=1
            else:
                n_trial_category[summary.iloc[n_trial].subject]=0
            current_frame = int(row.frame)
    else:
        final_sheet['subject'].append(summary.iloc[n_trial].subject)
        final_sheet['n_trial'].append(n_trial_category[summary.iloc[n_trial].subject])
        final_sheet['frame'].append(int(row.frame-current_frame))
        final_sheet['start_stop'].append(row.start_stop)

final_sheet = pd.DataFrame.from_dict(final_sheet)

sheet_64 = final_sheet[final_sheet.subject==32364]

vid_root = '/media/sebastian/MYLINUXLIVE/MT/tracking_cropped'

vids = sorted([f for f in os.listdir(vid_root) if '.avi'  in f and (int(f.split('.')[0].split('_')[-1]))%2])

print(len(vids))

data = h5py.File('/home/sebastian/Desktop/data.h5','w')
data.create_dataset("X", (1000,600,500,3), dtype='i')
data.create_dataset("Y", (1000,1), dtype='i')

