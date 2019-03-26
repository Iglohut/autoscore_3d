import os
#os.chdir('/media/deeplabchop/MD_Smits/Internship/autoscore_3d') # For at Marvin
print(os.getcwd())

import sys
sys.path.append("..") # Adds higher directory to python modules path
from PredictUtils.utils import im_equalize_hist
from keras.models import Model, load_model
import numpy as np
import h5py
import cv2
from pathlib import Path
from OS_utils import read_yaml
import skimage.io
import skvideo.io
import pandas as pd
os.chdir('/media/deeplabchop/MD_Smits/Internship/autoscore_3d') # For at Marvin

# Set GPU (For Marvin)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 


def autoscore_video(input_video, model_final, baseframe,  n_frames=9, n_frames_steps=3, skipfirst=0, skiplast=0):
    # Open video and auxiliary information
    reader = skvideo.io.vreader(input_video)
    # input_shape = tuple(np.shape(model_final.input)[2:4]) # Height and Width
#    video_shape = skvideo.io.FFmpegReader(input_video).getShape()
#    total_frames = video_shape[0]

    # One video loop
    frames = []
    labels = []
    timeseries = []
    for i, frame in enumerate(reader):

        # Some preprocessing
        frame = im_equalize_hist(baseframe, frame)
        #    frame = cv2.resize(frame, (512, 384))
        frame = cv2.resize(frame.astype("uint8"), (512, 384))
        #    if input_shape[0] and (not video_shape[1:3] == input_shape): cv2.resize(frame, input_shape)

        if True: #i > skipfirst and i < total_frames - skiplast:  # Predicter loop

            frames.append(frame)

            if len(frames) == n_frames * n_frames_steps:
                X = np.array([frames[slice(0, n_frames * n_frames_steps, n_frames_steps)]])
                Y = model_final.predict(X)

                if Y[0][0] < 0.5:  # Exploring decision
                    explore = 0
                else:
                    explore = 1
                timeseries.append(explore)

                # print("yey", i, Y)
                _ = frames.pop(0)
    reader.close()

    return timeseries

def save_timeseries(timeseries, input_video):
    """

    :param timeseries: vector of per frames decision "is exploring"
    :param input_video_relative: name of the video
    :return: nothing. Saves the timeseries
    """
    savepath = os.getcwd() + '/data/ehmt1_autoscores/' + input_video.split("/")[-1] + '.csv'
    df = pd.DataFrame(timeseries, columns =["Explore"])
    df.to_csv(savepath, index=False)


class DataRuler(object):
    """
    Helper class to work with the VideoNamesStatus.csv dataset
    """
    def __init__(self, VideoNamesSatusPath):
        self.VideoNamesStatusPath = VideoNamesSatusPath
        self.df = pd.read_csv(VideoNamesSatusPath)

        self.index = self.df[self.df.StatusPredicted == 0 ].index[0] # The index of the first vid that is not yet predicted

    def vidname(self):
        return self.df.iloc[self.index].VideoName

    def full_vidname(self):
        relative_path = self.vidname()
        vidname = os.getcwd().split("autoscore_3d")[0] + "Intellectual_Disability/Intellectual_Disability" + relative_path[1:]
        return vidname

    def status(self):
        return self.df.at[self.index, "StatusPredicted"]

    def update_status(self, new_status):
        self.df.at[self.index, "StatusPredicted"] = new_status
        self.df.to_csv(self.VideoNamesStatusPath, index=False)

    def goto_first_unprecited_video(self):
        old_index = self.index + 0
        self.index = self.df[self.df.StatusPredicted == 0].index[0]
        if self.index == old_index: print("Did not move to the next video. Did you update the status?")

    def next_video(self):
        self.index += 1

        while True:
            if self.index >= len(self.df):
                print("Error. Reached the last video.")
                # self.index -= 1
                break
            elif self.df.at[self.index, "StatusPredicted"] != 0: # To sometimes handmake status 2: later check if fps=0 go next
                self.index += 1

            elif self.df.at[self.index, "StatusPredicted"] == 0:
                break


    def gotovideo(self, vidname):
        realtive_path = "." + vidname[vidname.find("/round"):]
        self.index = self.df[self.df.VideoName == realtive_path].index[0]


    def cantinue(self):
        return self.index <= (len(self.df) -1)

# Setting parameters
n_frames = 9
n_frames_steps = 3

# Get config file
config = read_yaml(Path(os.getcwd()).resolve() / 'config.yaml')

# Load model
# model_final = load_model("/media/iglohut/MD_Smits/Internship/autoscore_3d/project/henk_bigger_3frames_checkpoint")
#model_final = load_model("/media/iglohut/MD_Smits/Internship/autoscore_3d/project/nepmodel_checkpoint")
model_final = load_model("/home/deeplabchop/src/autoscore_iglo2/project/henk_bigger_3frames_checkpoint")
print("loaded model")

# Load baseframe
baseframe = skimage.io.imread(str(Path(os.getcwd()).resolve()) + "/PredictUtils/baseframe.png")


# # Input video
# input_video = "/media/iglohut/MD_Smits/Internship/Intellectual_Disability/Intellectual_Disability/round_8/mouse_training_OS_5trials_inteldis_36_44_or_101_9/mouse_training_OS_5trials_inteldis_36_44_or_101_9_t0001_raw.avi"
#

# Load status data
VideoNamesSatusPath = Path("./PredictUtils/VideoNamesStatus.csv").resolve()
# df = pd.read_csv(VideoNamesSatusPath)

VideoSet = DataRuler(VideoNamesSatusPath)


while VideoSet.cantinue(): # If not at end of datafile
    input_video = VideoSet.full_vidname()
    print("At video:", VideoSet.index, input_video)
    timeseries = autoscore_video(input_video, model_final, baseframe=baseframe, n_frames=n_frames, n_frames_steps=n_frames_steps)

    # Some saving stuff timeseries later...
    save_timeseries(timeseries, input_video)

    # Update status and go to next video
    VideoSet.update_status(new_status = 1)
    VideoSet.next_video()




#
# VideoSet.index = 0
# while VideoSet.cantinue():
#     VideoSet.update_status(new_status=0)
#     # VideoSet.next_video()
#     VideoSet.index += 1
#     print(VideoSet.index)
#
#
