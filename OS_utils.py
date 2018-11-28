import pandas as pd
import os
import h5py
import skvideo.io
import numpy as np
import yaml
from pathlib import Path
import cv2
import random
import csv

# For csv
def read_yaml(yaml_path):
    if not yaml_path.exists():
        raise FileNotFoundError('No such file or directory: {}'.format(yaml_path))

    yd = dict()
    with open(yaml_path, 'r') as yf:
        yd.update(yaml.load(yf))

    return yd

def get_csv_paths(video_path):
    """
    :param video_path: full path to where one round of videos is stored
    :return: paths to the log file and the schema file
    """
    for f in os.listdir(video_path):
        if 'csv' in f:
            if 'log' in f and 'lock' not in f:
                log_path = video_path / f

            elif 'log' not in f and 'OS' in f and 'lock' not in f:
                scheme_path = video_path / f
    return log_path, scheme_path

def make_summary_csv(log_path, scheme_path):
    """
    Makes a summary sheet containing frame, trial, start_stop as columns containing pure explorations

    :param log_path: path to log file of round
    :param scheme_path: path to scheme file of round
    :return:
    """
    scheme = pd.read_csv(scheme_path)
    data = pd.read_csv(log_path)

    final_sheet = {'run_nr': [], 'subject': [], 'n_trial': [], 'frame': [], 'start_stop': []} # TODO add start_rec_frame: to not search below later!

    cur_run = 0
    cur_frame = 0
    for i, row in data.iterrows():

        if row.type == 'TR' and row.start_stop == True:
            cur_run += 1  # Every TR is new trial (in sequence/run)

        elif row.type == 'TR' and row.start_stop == False:  # This is when trial ends
            pass

        else:  # Only take real explorations
            final_sheet['run_nr'].append(scheme.iloc[cur_run - 1].run_nr)
            final_sheet['subject'].append(scheme.iloc[cur_run - 1].subject)
            final_sheet['n_trial'].append(scheme.iloc[cur_run - 1].trial)
            final_sheet['frame'].append(int(row.frame - cur_frame))  # was - current frame
            final_sheet['start_stop'].append(row.start_stop)

    final_sheet = pd.DataFrame.from_dict(final_sheet)
    final_sheet.to_csv(video_path / 'summary_sheet.csv')

def get_summary_file(video_path):
    """
    Gets the summary_sheet.csv file. If it does not exist it will try to make one.

    :param video_path:
    :return:
    """
    summarypath = video_path / 'summary_sheet.csv'
    if not summarypath.exists():
        log_path, scheme_path = get_csv_paths(video_path)
        make_summary_csv(log_path, scheme_path)

    if summarypath.exists():
        return pd.read_csv(summarypath)
    else:
        print("The path summary_sheet.csv doesn't exist! Looking in:", summarypath)
        print("\nMake sure the videopath contains a log and schema .csv file")


# For videos
def get_nframes(video_name, last_frame, nframes = 10):
    """
    Extracts nframes from video to create training batches.

    :param video_name: absolute path to video
    :param last_frame: frame of interest
    :param nframes: number of frames to take
    :return: array of size nframes x width x height x RGB, with last frame being the last frame
    """
    from_frame = last_frame - nframes

    cap = cv2.VideoCapture(str(video_name))
    cap.set(1, from_frame)
    frames = [cap.read()[1] for i in range(nframes)]
    framesarray = np.asarray(frames)
    return framesarray

def _showvid(video_name, from_frame = 0):
    """
    HELPER FUNCTION
    Navigate through video frame by frame from a certain frame.

    :param video_name: videopath
    :param from_frame: show vid from this frame
    :return: nothing
    """
    cap = cv2.VideoCapture(str(video_name))
    cap.set(1, from_frame)
    for i in range(from_frame , from_frame + 999999):
        ret, frame = cap.read()
        if not ret:
            print("Grab frame unsuccessful. ABORT MISSION!")
            break
        cv2.imshow('frame: ' + str(i), frame)
        # Set waitKey
        key = cv2.waitKey()
        if key == ord('q'):
            break
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

def _vidlength(video_name):
    """

    :param video_name: path to video
    :return: number of frames in video
    """
    cap = cv2.VideoCapture(str(video_name))
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    length = int(cv2.VideoCapture.get(cap, property_id))
    return length

def _getall_vidlengths(video_path):
    """
    Gets length of all videos. stored on alphabetical order!
    :param video_path: path to al videos
    :return:
    """
    all_videos = get_video_paths(video_path)
    all_video_lengths = [_vidlength(v) for v in all_videos]
    return np.asarray(all_video_lengths)

def get_video_paths(video_path):
    """
    :param video_path: path to where video is stored
    :return: list of all the videos on alphabetical order
    """
    videos = []
    for f in sorted(os.listdir(video_path)):
        if '.avi' in f:
            videos.append(str(video_path / f))
    return videos


def get_slices(data, project_path, model_name, n_frames=9, val_split=0.9, equal_labels=1, n_stacker=1, steps = 1):
    """
    Gets the slice indices of the data that are used for training and validation.
    """
    def midlabel(y, n_frames):
        return y[-int(np.ceil(n_frames / 2))]

    slice_path = str(project_path) + model_name + '_slices.pkl'
    if not Path(slice_path).exists():
        print("Making training and validation set in", slice_path)
        sets = [slice(0 + i * n_stacker, n_stacker * i + n_frames * steps, steps) for i in
                range(int((data['X'].shape[0] - n_frames) / n_stacker))]  # stacked windows of t_size

        if equal_labels:
            labels = [midlabel(data['Y'][sets[i]], n_frames) for i in range(len(sets))]
            df = pd.DataFrame({'slices': sets,
                               'labels': labels})
            df_1 = df.loc[df['labels'] == 1]
            df_0 = df.loc[df['labels'] == 0]
            df_0 = df_0.sample(n=len(df_1), replace=False) # Because label 1 is least common
            df = df_1.append(df_0)
            sets = list(df['slices'].values)


        slices_x = np.random.permutation(sets)
        slices_train = slices_x[0: int(val_split * len(slices_x))]
        slices_val = slices_x[int(val_split * len(slices_x)):]

        df_train = pd.DataFrame(slices_train, columns=['slices_train'])
        df_val = pd.DataFrame(slices_val, columns=['slices_val'])
        df = pd.concat([df_train, df_val], axis=1)
        df.to_pickle(slice_path)

    # Read existing slices
    slices = pd.read_pickle(slice_path)

    slices_val = slices['slices_val']
    slices_val = slices_val.dropna().values

    slices_train = slices['slices_train']
    slices_train = slices_train.dropna().values

    return slices_train, slices_val


class Logger(object):
    def __init__(self, project_path, model_name):
        self.project_path = project_path
        self.model_name = model_name
        self.logpath = Path(self.project_path + self.model_name + '_metrics_logger.csv').resolve()

        if not self.logpath.exists():
            print("Making a new metrics logfile in", str(self.logpath))
            data = {'Epoch': [], 'train_loss': [], 'train mae': [], 'train acc': [], 'train auc': [], 'val_loss': [],
                    'val mae': [], 'val acc': [], 'val auc': []}
            df = pd.DataFrame.from_dict(data)
            df.to_csv(self.logpath, index=False)

            self.start_epoch = 1
        else:
            df = pd.read_csv(self.logpath)
            self.start_epoch = len(df['Epoch']) + 1


    def store(self, epoch, train_loss, train_mae, train_acc, train_auc, val_loss, val_mae, val_acc, val_auc):
        f = open(self.logpath, 'a')
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_mae, train_acc, train_auc, val_loss, val_mae, val_acc, val_auc])
        f.close()




#
#
## Set the paths
#config_path = Path(os.getcwd()).resolve()
#config = read_yaml(config_path / 'config.yaml')
#video_path = Path(config['video_set'])
#log_path, scheme_path = get_csv_paths(video_path)
#
## Set parameters
#nframes = config['model_params']['n_frames'] # Number of frames to read
#video_resize_width = config['resize']['width']
#video_resize_height = config['resize']['height']
#
#
## # Make the summary_sheet.csv
## make_summary_csv(log_path, scheme_path) # This doesn't need to be called in later versions. get_summary_file does this.
#
#
## Video stuff
#video_name  = video_path / 'mouse_training_OS_5trials_inteldis_32_35_t0001_raw.avi'
#
#round_summary = get_summary_file(video_path)
#
## runs_range = range(int(round_summary.run_nr.min()),int(round_summary.run_nr.max()))
#
#all_videos = get_video_paths(video_path)
#
## This should go in a loop
#current_video = random.choice(all_videos)
#current_run = int(current_video.split('t')[-1].split('_')[0])
#
#
#current_runsummary = round_summary.loc[round_summary['run_nr'] == current_run] # All info from current video
#
#
#
#
#
#_vidlength(video_name) # TODO !! Change make_summary to deduct vidlength of all previous videos from current frame.. download all videos first :(
#_vidlength(video_path / 'mouse_training_OS_5trials_inteldis_32_35_t0002_raw.avi')
#_vidlength(video_path / 'mouse_training_OS_5trials_inteldis_32_35_t0003_raw.avi')
#_vidlength(video_path / 'mouse_training_OS_5trials_inteldis_32_35_t0004_raw.avi')
#
#
#
#
## TODO 1. Open random videos and create nvid random segments of 10 frames with y label for last frame: ((n, 10, x, y, 3), (n))
### TODO 1.1. Store in h5 file: on usb stick
#### TODO 1.2. Remember to read final_sheet and match... Remember maybe loop over folder of rounds1-8 randomly too OR do it per round slowly as my laptop can't handle much anyway
##### TODO 1.2.1. Balance classes yes/no
## TODO Make own generator that access h5 lines directly and not load entire file
#
## vids = sorted([f for f in os.listdir(vid_root) if '.avi' in f and (int(f.split('.')[0].split('_')[-1].split('t')[-1]))%2]) # Do this withotu the splti check
##
##
## print(len(vids))
##
## videodata = [skvideo.io.vread(os.path.join(vid_root,vids[i])) for i in range(1)]
## videodata = videodata[0][:,:,:512,:]
## scoremat = np.zeros((videodata.shape[0],))
##
## for i,row in sheet_64.iterrows():
##     if row.n_trial>0:
##         break
##     if row.start_stop:
##         start = row.frame
##     else:
##         scoremat[start:row.frame]=True
##         videodata[start:row.frame,0:10,0:10,:]=0
##
## data = h5py.File('/home/iglohut/Documents/MemDyn/OS_Data/data.h5','w')
## data['X']=videodata
## data['Y']=scoremat
## data.close()
#
