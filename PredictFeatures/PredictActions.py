import pandas as pd
import numpy as np
import os
import cv2
from PredictFeatures.VisualizeBox import BoxTemplate

class SequenceExtracter():
    df_all = pd.read_csv('./data/ehmt1/VideoNamesStatus.csv')
    def __init__(self, vidnumber):
        self.df = self.df_all.iloc[vidnumber]

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
        delta_HD = self.template.df["Trial_flip"]["Trial_flip"]["Degrees"].values[0] * (np.pi / 2) # Some videos are flipped
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
        for pivot in self.ActionSequence["pivot_locations"]:
            action = None
            if pivot is not None:
                superlocation = pivot[0]
                sublocation = pivot[1]

                if superlocation == "Wall":
                    action = "Wall"

                if superlocation == "Object":
                    if sublocation == self.df.obj_1:
                        action = "obj_1"
                    if sublocation == self.df.obj_2:
                        action == "obj_2"

                if superlocation == "Corner":
                    action = "Corner"

            actions.append(action)
        self.ActionSequence["actions"] = actions

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


class IconFrame():
    def __init__(self, frame):
        self.frame = self._load_frame(frame)
        self.new_frame = self.frame.copy()
        self.wall = cv2.resize(cv2.imread(self.wall_path, -1), self.goalsize)
        self.corner = cv2.resize(cv2.imread(self.corner_path, -1), self.goalsize)
        self.explore = cv2.resize(cv2.imread(self.explore_path, -1), self.goalsize)

    @property
    def wall_path(self):
        return os.getcwd() + "/PredictFeatures/icons/trumpwall.png"

    @property
    def corner_path(self):
        return os.getcwd() + "/PredictFeatures/icons/corner.png"

    @property
    def explore_path(self):
        return os.getcwd() + "/PredictFeatures/icons/explore.png"

    @property
    def goalsize(self):
        sizereduction = 7
        dimensions = np.array(np.array(self.frame.shape) / sizereduction).astype(np.int)
        return (dimensions[1], dimensions[0])


    def _load_frame(self, frame):
        if len(frame.shape) == 3:
            return frame
        else:
            return cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

    def embed_icon(self, overlay, x_offset, y_offset, chosen = 0):
        """Embed single icon on frame"""
        y1, y2 = y_offset, y_offset + overlay.shape[0]
        x1, x2 = x_offset, x_offset + overlay.shape[1]


        if chosen == 0: alpha = 243
        if chosen == 1: alpha =255
        alpha_s = overlay[:, :, 3] / alpha
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            self.new_frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                      alpha_l * self.new_frame[y1:y2, x1:x2, c])

    def embed_icons(self, action = None):
        """embeds all icons on frame, but only makes the one that has an action salient."""
        dimensions = np.array(np.array(self.frame.shape))[0:2]

        chosen_wall = 0
        chosen_object = 0
        chosen_corner = 0
        if action == "Wall":
            chosen_wall = 1
        if action in ["Object", "obj_1", "obj_2"]:
            chosen_object = 1
        if action == "Corner":
            chosen_corner = 1

        y_loc = int(dimensions[0] * 0.01)
        self.embed_icon(self.wall, int(dimensions[1] * 0.3), y_loc, chosen_wall)
        self.embed_icon(self.corner, int(dimensions[1] * 0.15), y_loc, chosen_corner)
        self.embed_icon(self.explore, int(dimensions[1] * 0.01), y_loc, chosen_object)


    def __call__(self):
        return self.new_frame

myvid = SequenceExtracter(1)


myframe = IconFrame(myvid.template.midframe)



myframe.embed_icons("Object")
cv2.imshow('Templateee', myframe())
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO automatic vidoemaker with limbs, actions ;; use parts of OrientationLabeler