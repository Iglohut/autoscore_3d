import pandas as pd
import numpy as np
import os
import cv2
from PredictFeatures.VisualizeBox import BoxTemplate
from PredictFeatures.MakeVideo import *

class SequenceExtracter():
    df_all = pd.read_csv('./data/ehmt1/VideoNamesStatus.csv')
    def __init__(self, vidnumber):
        self.df = self.df_all.iloc[vidnumber] # Dataframe with trial information

        if self.canalyze:
            self.template = BoxTemplate(self.df.VideoName) # Get the BoxTemplate
            self.df_pose = pd.read_csv(self.posefile, header=[0, 1], skipinitialspace=True)
        else:
            print("My apologies, this video was not analyzed by autoscore and DLC. Video({}):".format(vidnumber), self.df.VideoName)

        self.ActionSequence = {
            "pivot_locations": [],
            "autoscores": [],
            "actions": []
        }

    @property
    def canalyze(self):
        """returns ff the video was actually analyzed according to file"""
        if self.df.StatusPredicted == 1:
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

    def headPoint(self, frame_idx):
        """
        :param frame_idx: index of frame in video/posefile
        :return: tuple pixel position of head
        """
        x = int(self.df_pose.loc[frame_idx, [("Nose", "x"), ("Left ear", "x"), ("Right  ear", "x")]].mean())
        y = int(self.df_pose.loc[frame_idx, [("Nose", "y"), ("Left ear", "y"), ("Right  ear", "y")]].mean())
        return (x, y)

    def headDirection(self, frame_idx):
        """
        Automatically controls for videos that are rotated
        :param frame_idx: index of frame in video/posefile
        :return: head direction in radians
        """
        HD = self.df_pose.loc[frame_idx, ("angle1", "Nose")]
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
        pivot_locations = []
        for i in range(len(self)): # for all estimated frames
            position = self.headPoint(i)
            location = self.template.detect(position)
            pivot_locations.append(location)

        self.ActionSequence["pivot_locations"] = pivot_locations


    def get_actions(self):
        """
        Calculated the actions based on DLC location, template, and autoscore (to be implemented).
        """
        if len(self.ActionSequence["pivot_locations"]) == 0:
            print("There's no pivot locations yet.")
            pass


        actions = []
        for i, pivot in enumerate(self.ActionSequence["pivot_locations"]):
            action = None
            if pivot is not None:
                superlocation = pivot[0]
                sublocation = pivot[1]

                if superlocation == "Wall" and self._check_wall(sublocation, i):
                    action = "Wall"

                if superlocation == "Object": # TODO combine with autoscore
                    if sublocation == self.df.obj_1:
                        action = "obj_1"
                    if sublocation == self.df.obj_2:
                        action = "obj_2"

                if superlocation == "Corner":
                    action = "Corner"

            actions.append(action)
        self.ActionSequence["actions"] = actions


    def _check_wall(self, sublocation, frame_idx):
        HD = self.headDirection(frame_idx)
        if sublocation == "North" and HD > 1 * np.pi / 10  and HD < 9 * np.pi / 10:
            return True
        elif sublocation == "East" and bool(HD < 4 * np.pi / 10 and not HD < -4 * np.pi / 10) ^ bool(HD > - 4 * np.pi / 10 and not HD > 4 * np.pi / 10): # ^ is XOR
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


myvid = SequenceExtracter(3093) # 2530 round8, 1670 round 7 norot,


# myframe = IconFrame(myvid.template.midframe)

# myframe.embed_icons()
# cv2.imshow('Templateee', myframe())
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Make video
myvid.get_pivot_locations()
myvid.get_actions()
myvid.make_video()

