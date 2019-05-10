import pandas as pd
import numpy as np
import os
import cv2
from PredictFeatures.VisualizeBox import BoxTemplate
from PredictFeatures.MakeVideo import *
import csv
import math

class SequenceExtracter:
    df_all = pd.read_csv('./data/ehmt1/VideoNamesStatus.csv')
    def __init__(self, vidnumber):
        self.df = SequenceExtracter.df_all.iloc[vidnumber] # Dataframe with trial information
        self.vidnumber = vidnumber

        # The action list for all frames
        self.ActionSequence = {
            "pivot_locations": [],
            "autoscores": [],
            "actions": []
        }


        if self.canalyze:
            self.template = BoxTemplate(self.df.VideoName) # Get the BoxTemplate
            self.df_pose = pd.read_csv(self.posefile, header=[0, 1], skipinitialspace=True)

            self.headPoints = self.calc_headPoints # precalculate all headpoints to speed up loops
            self.headDirections = list(self.df_pose[("angle1" , "Nose")])
            self.ActionSequence['autoscores'] = self.set_autoscores


        else:
            print("My apologies, this video was not analyzed by autoscore and DLC OR is already fully analyzed!. Video({}):".format(vidnumber), self.df.VideoName)

    @property
    def canalyze(self):
        """returns ff the video was actually analyzed according to file"""
        if self.df.StatusPredicted == 1 and not math.isnan(self.df.genotype): # If analyzed AND train data available
            return True
        else:
            return False

    @property
    def vidname(self):
        """
        :return: raw path of current video
        """
        return self.template.vidname

    @property
    def posefile(self):
        """Returns the pose estimation absolute path of the current video"""
        posedir = os.getcwd() + '/data/ehmt1/ehmt1_poses'
        posefiles = os.listdir(posedir)
        posefiles = [file for file in posefiles if 'ORI' in file]
        myposefile = [file for file in posefiles if self.df.VideoName.split('/')[-1].split('.')[0] in file]

        myposefile = posedir + "/" + myposefile[0]
        return myposefile

    @property
    def autoscorefile(self):
        """Returns thge autoscore estimation path of the current video"""
        dir = os.getcwd() + '/data/ehmt1/ehmt1_autoscores'
        files = os.listdir(dir)
        myfile = [file for file in files if self.df.VideoName.split('/')[-1].split('.')[0] in file if '#' not in file]
        myfile = dir + "/" + myfile[0]
        return myfile

    @property
    def set_autoscores(self):
        """Returns for all frames decision exploring object or not"""
        df = pd.read_csv(self.autoscorefile)
        df = list(df["Explore"])
        df = list(np.zeros(13)) + df + list(np.zeros(14))# Because autoscore used windows of 27frames
        return df

    @property
    def calc_headPoints(self):
        xs = self.df_pose[[("Nose", "x"), ("Left ear", "x"), ("Right  ear", "x")]].mean(axis=1)
        ys = self.df_pose[[("Nose", "y"), ("Left ear", "y"), ("Right  ear", "y")]].mean(axis=1)

        return {'x': list(xs),
                'y': list(ys)
                }

    def headPoint(self, frame_idx):
        """
        :param frame_idx: index of frame in video/posefile
        :return: tuple pixel position of head
        """
        x = self.headPoints['x'][frame_idx]
        y = self.headPoints['y'][frame_idx]
        return (x, y)

    def headDirection(self, frame_idx):
        """
        Automatically controls for videos that are rotated
        :param frame_idx: index of frame in video/posefile
        :return: head direction in radians
        """
        # HD = self.df_pose.loc[frame_idx, ("angle1", "Nose")]
        HD = self.headDirections[frame_idx]
        flips = self.template.df["Trial_flip"]["Trial_flip"]["Degrees"].values[0] # Clockwise flips
        if np.sign(HD) == -1:
            HD += 2 * np.pi
        if flips > 0:
            delta_HD =  (4 - flips) * (np.pi / 2) # Some videos are flipped
        else:
            delta_HD = 0
        HD = np.angle(np.exp(1j * (HD + delta_HD)))
        return HD


    def get_pivot_locations(self):
        """
        Creates sequence of all pivot locations where the mouse was for all frames.
        """
        if not self.canalyze:
            return

        pivot_locations = []
        for i in range(len(self)): # for all estimated frames
            # print("Get pivot locations: {}/{}".format(i, len(self)))
            position = self.headPoint(i)
            location = self.template.detect(position, self.ActionSequence["autoscores"][i])
            pivot_locations.append(location)

        self.ActionSequence["pivot_locations"] = pivot_locations


    def get_actions(self):
        """
        Calculates the actions based on DLC location, template, and autoscore (to be implemented).
        """
        if len(self.ActionSequence["pivot_locations"]) == 0:
            print("There's no pivot locations yet.")
            return

        all_actions = []
        for i, pivots in enumerate(self.ActionSequence["pivot_locations"]):
            frame_actions = []
            for pivot in pivots:
                if pivot is not None:
                    superlocation = pivot[0]
                    sublocation = pivot[1]

                    if superlocation == "Wall" and self._check_wall(sublocation, i):
                        frame_actions.append("Wall")

                    if superlocation == "Object":
                        if sublocation == self.df.obj_1:
                            frame_actions.append('obj_1')
                        if sublocation == self.df.obj_2:
                            frame_actions.append('obj_2')

                    if superlocation == "Corner":
                        frame_actions.append('Corner')

            if len(frame_actions) == 0: frame_actions = [None]
            all_actions.append(frame_actions)
        self.ActionSequence["actions"] = all_actions



    def _check_wall(self, sublocation, frame_idx):
        HD = self.headDirection(frame_idx)
        if sublocation == "North" and HD > 1 * np.pi / 10  and HD < 9 * np.pi / 10:
            return True
        elif sublocation == "East" and bool(HD < 4 * np.pi / 10 and not HD < -4 * np.pi / 10): # ^ is XOR
            return True
        elif sublocation == "South" and HD < -1 * np.pi / 10 and HD > - 9 * np.pi / 10:
            return True
        elif sublocation == "West" and bool(HD < - 6 * np.pi / 10) ^ bool(HD > 6 * np.pi / 10): # ^ is XOR
            return True
        else:
            return False

    def make_video(self):
        if bool(len(self)) and bool(len(self.ActionSequence["actions"])):
            make_video(self)
        else:
            print("Cannot make video. Either no actions yet or ambiguous videolength.")


    def save_actions(self, path=None):
        if path is None:
            path = os.getcwd() + '/data/ehmt1/ehmt1_actions/' + self.vidname.split('.')[0].split('/')[-1] + '.csv'

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.ActionSequence['actions'])

    def save_status(self, status=3):
        SequenceExtracter.df_all.loc[self.vidnumber, 'StatusPredicted'] = status
        SequenceExtracter.df_all.to_csv('./data/ehmt1/VideoNamesStatus.csv', index=False)
        print("Successfully updated the status({}) of {}".format(status, self.vidname))

    def __call__(self):
        return self.ActionSequence

    def __len__(self):
        df_length = len(self.df_pose)
        videolength = self.df.framelength

        if df_length == videolength:
            return df_length
        elif df_length > videolength:
            print("Couldn't determine length. More pose estimations than frames.")
            return None

        elif videolength > df_length:
            print("Couldn't determine length. More frames than estimated poses.")
            return None


# myvid = SequenceExtracter(760) # 2530 round8, 1670 round 7 norot, --2701 examplevid


# myframe = IconFrame(myvid.template.midframe)

# myframe.embed_icons()
# cv2.imshow('Templateee', myframe())
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Make video
# myvid.get_pivot_locations()
# myvid.get_actions()
# # myvid.make_video()
# myvid.save_actions()
# for i in range(len(myvid.ActionSequence["actions"])):
#     print(myvid.ActionSequence["actions"][i], i, myvid.ActionSequence["pivot_locations"][i], myvid.headPoint(i))


# listpath = '/media/iglohut/MD_Smits/Internship/autoscore_3d/data/ehmt1/ehmt1_actions/mouse_training_OS_5trials_inteldis_1_7animals_t0002_raw.csv'
# with open(listpath, 'r') as f:
#     reader = csv.reader(f)
#     mylist = list(reader)

# TODO make class that iterates over all len(SeuenceExtracter.df_all) to analyze and save the data
# TODO - make that multiprocessing

# fails = []
# for i in range(len(SequenceExtracter.df_all)):
#
#     myvid = SequenceExtracter(i)
#
#     if myvid.canalyze:
#         print("Analyzing video({}): {}".format(i, myvid.vidname))
#         myvid.get_pivot_locations()
#         myvid.get_actions()
#         myvid.save_actions()
#         myvid.save_status(status=3)
#     else:
#         print("Couldn't analyze video({})".format(i))
#         fails.append(i)