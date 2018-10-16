import pandas as pd
import os
import h5py
import skvideo.io
import numpy as np

summary = pd.read_csv('/home/iglohut/Documents/MemDyn/OS_Data/CSVFiles/mouse_training_OS_5trials_inteldis_8_12animals.csv')

data = pd.read_csv('/home/iglohut/Documents/MemDyn/OS_Data/CSVFiles/mouse_training_OS_5trials_inteldis_8_12animals_log.csv')


final_sheet = {'subject': [], 'n_trial': [], 'frame': [], 'start_stop': []}

n_trial = 0
current_frame = 0
n_trial_category = {}
for i, row in data.iterrows():

    if row.type == 'TR':
        if row.start_stop == False:
            n_trial += 1

        else:
            if summary.iloc[n_trial].subject in n_trial_category.keys():
                n_trial_category[summary.iloc[n_trial].subject] += 1
            else:
                n_trial_category[summary.iloc[n_trial].subject] = 0
            current_frame = int(row.frame)
    else:
        final_sheet['subject'].append(summary.iloc[n_trial].subject)
        final_sheet['n_trial'].append(n_trial_category[summary.iloc[n_trial].subject])
        final_sheet['frame'].append(int(row.frame - current_frame))
        final_sheet['start_stop'].append(row.start_stop)

final_sheet = pd.DataFrame.from_dict(final_sheet)
final_sheet.to_csv('/home/iglohut/Documents/MemDyn/OS_Data/CSVFiles/final_sheet.csv')




vid_root = '/home/iglohut/Documents/MemDyn/OS_Data/Videos'

vids = sorted([f for f in os.listdir(vid_root) if '.avi' in f and (int(f.split('.')[0].split('_')[-1].split('t')[-1]))%2]) # Do this withotu the splti check


print(len(vids))

videodata = [skvideo.io.vread(os.path.join(vid_root,vids[i])) for i in range(1)]
videodata = videodata[0][:,:,:512,:]
scoremat = np.zeros((videodata.shape[0],))

for i,row in sheet_64.iterrows():
    if row.n_trial>0:
        break
    if row.start_stop:
        start = row.frame
    else:
        scoremat[start:row.frame]=True
        videodata[start:row.frame,0:10,0:10,:]=0

data = h5py.File('/home/iglohut/Documents/MemDyn/OS_Data/data.h5','w')
data['X']=videodata
data['Y']=scoremat
data.close()

