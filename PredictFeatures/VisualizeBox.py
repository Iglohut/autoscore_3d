import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import PredictFeatures.ShapeHelper as PS
import cv2

df = pd.read_csv('./data/ehmt1/VideoNamesStatus.csv')
df_boxloc = pd.read_csv('./data/ehmt1/BoxLocations.csv')

from sklearn.cluster import KMeans
from collections import Counter
import cv2  # for resizing image


def get_dominant_color(image, k=4, image_processing_size=None):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)

class BoxTemplate:
    df_boxloc = pd.read_csv('./data/ehmt1/BoxLocations.csv', header=[0, 1, 2])
    def __init__(self, video_path):
        self.vidname = os.getcwd().split("autoscore_3d")[0] + "Intellectual_Disability/Intellectual_Disability" + video_path[1:]
        self.round = int(video_path[video_path.find('round')+6]) # Round number
        self.trial = int(myvid.split("_t0")[-1].split("_")[0])  # Trial number - because the camera flip...
        self._videodimension() # Set dimension of box
        self._grab_midframe() # Get a frame
        self._boxcolor()

        self.df = self.df_boxloc.loc[self.df_boxloc["Round"]["Round"]["Round"] == self.round] # Select BoxLocations of this round
        self.df = self.df[self.df["BoxColor"]["BoxColor"]["BoxColor"] == self.boxcolor]
        if self.round == 7 and self.trial >= 21:
            self.df = self.df[self.df["Trial_flip"]["Trial_flip"]["Trial_flip"] == 21]
        if self.round == 7 and self.trial < 21:
            self.df = self.df[self.df["Trial_flip"]["Trial_flip"]["Trial_flip"] == 0] # TODO Flip is in the first 21 trials: 90deg.. delete trials, too hard to control for?


        self._set_locs() # Set per location the radius of detection --> self.full_locations

    def _videodimension(self):
        """Get Dimension of current video (trial)."""
        cap = cv2.VideoCapture(self.vidname)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        cap.release()

    def _grab_midframe(self):
        "Grabs middle frame for illustration purposes"
        cap = cv2.VideoCapture(self.vidname)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(1, int(frameCount / 2))
        ret, frame = cap.read()
        GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cap.release()
        self.midframe = GRAY

    def _boxcolor(self):
        "Identifies if box is green or white"
        cap = cv2.VideoCapture(self.vidname)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(1, int(frameCount / 2))
        ret, frame = cap.read()
        clt = get_dominant_color(frame)
        if clt[1] > np.mean(clt) + np.std(clt):
            color = "Green"
        else:
            color = "White"
        self.boxcolor = color
        cap.release()


    def _set_locs(self):
        """Sets, for pivot each location, the paramaters for its shape where something is detected.
        self.full_locations"""
        locations = self.df.columns.values[3::2].tolist() # Select columns object/corner/wall and sublocations to iterate over
        self.full_locations = []
        for location in locations: # For every possible mapped location
            superlocation = location[0] # Object/Corner/Wall
            sublocation = location[1] # Left/West etc
            area_position = self.df[superlocation][sublocation]  # x,y coordinate

            if superlocation == "Object":
                radius = int(np.linalg.norm(area_position - self.df["Corner"][sublocation]) / 2)
            elif superlocation == "Corner":
                radius = int(np.linalg.norm(area_position - self.df["Object"][sublocation]) / 2)
            elif superlocation == "Wall": # Wall are represented as rectangles!
                wall_length = 4
                wall_width = 3
                sidewall_extension = 3
                if sublocation == "North":
                    dx =abs(self.df["Corner"]["UR"]['x'].values[0] - self.df["Corner"]["UL"]['x'].values[0])

                    x1 = self.df["Corner"]["UL"]['x'].values[0] + dx / wall_length
                    x2 = self.df["Corner"]["UR"]['x'].values[0] - dx / wall_length

                    dy = abs((self.df["Corner"]["UR"]['y'].values[0] - self.df["Object"]["UL"]['y'].values[0]))
                    y1 = self.df["Corner"]["UR"]['y'].values[0] + dy / wall_width
                    y2 = self.df["Corner"]["UR"]['y'].values[0] - dy / wall_width

                elif sublocation == "South":
                    dx = abs(self.df["Corner"]["LR"]['x'].values[0] - self.df["Corner"]["LL"]['x'].values[0])

                    x1 = self.df["Corner"]["LL"]['x'].values[0] + dx / wall_length
                    x2 = self.df["Corner"]["LR"]['x'].values[0] - dx / wall_length

                    dy = abs((self.df["Corner"]["LR"]['y'].values[0] - self.df["Object"]["LL"]['y'].values[0]))
                    y1 = self.df["Corner"]["LR"]['y'].values[0] + dy / wall_width
                    y2 = self.df["Corner"]["LR"]['y'].values[0] - dy / wall_width

                elif sublocation == "East":
                    wall_length = ((self.height / self.width) * wall_length)
                    dx = abs(self.df["Corner"]["LR"]['x'].values[0] - self.df["Object"]["LR"]['x'].values[0])

                    x1 = self.df["Corner"]["LR"]['x'].values[0] + sidewall_extension * dx / wall_width
                    x2 = self.df["Corner"]["LR"]['x'].values[0] - dx / wall_width

                    dy = abs(self.df["Corner"]["LR"]['y'].values[0] - self.df["Object"]["UR"]['y'].values[0])
                    y1 = self.df["Corner"]["LR"]['y'].values[0] - dy / wall_length
                    y2 = self.df["Corner"]["UR"]['y'].values[0] + dy / wall_length

                elif sublocation == "West":
                    wall_length = ((self.height / self.width) * wall_length)
                    dx = abs(self.df["Corner"]["LL"]['x'].values[0] - self.df["Object"]["LL"]['x'].values[0])

                    x1 = self.df["Corner"]["LL"]['x'].values[0] + dx / wall_width
                    x2 = self.df["Corner"]["LL"]['x'].values[0] - sidewall_extension * dx / wall_width

                    dy = abs(self.df["Corner"]["LL"]['y'].values[0] - self.df["Object"]["UL"]['y'].values[0])
                    y1 = self.df["Corner"]["LL"]['y'].values[0] - dy / wall_length
                    y2 = self.df["Corner"]["UL"]['y'].values[0] + dy / wall_length

                point_low = [x1, y1]
                point_high = [x2, y2]
                points =[]
                for point in [point_low, point_high]:
                    x = point[0]
                    y = point[1]
                    if x < 0:
                        x = 0
                    elif x > self.width:
                        x = self.width
                    elif y < 0:
                        y = 0
                    elif y > self.height:
                        y = self.height

                    points.append(PS.Point(int(x), int(y)))
                radius = PS.Rect(points[0], points[1]) # The rectangle

            self.full_locations.append((superlocation, sublocation, radius))

    def closest(self, position):
        """
        :param position: tuple (x, y) position
        :return: list or name/distance of location closest to
        """
        pass
        # locations = self.df.columns.values[2::2].tolist() # Select columns object/corner/wall and sublocations to iterate over
        #
        # distances = []
        # for location in locations: # For every possible mapped location
        #     superlocation = location[0] # Object/Corner/Wall
        #     sublocation = location[1] # Left/West etc
        #     area_position = self.df[superlocation][sublocation] # x,y coordinates
        #
        #     distance = np.linalg.norm(area_position - position) # L2 norm distance
        #
        #
        #     # This sets a radius for both the object and corner
        #
        #
        #
        #     distances.append((distance, superlocation, sublocation))
        #
        # return distances

    def detect(self, position):
        """

        :param position: tuple (x, y) of pixel position of animal
        :return: the pivot point in which the animal is (if any)
        """
        pass
    # IF in any radius return

    def template(self):
        "Image template of the box"
        frame = self.midframe
        overlay = frame.copy()
        output = frame.copy()
        alpha = 0.8

        for location in self.full_locations:
            superlocation = location[0]
            sublocation = location[1]
            area_position = tuple(self.df[superlocation][sublocation].values[0])  # x,y coordinates

            if superlocation == "Wall":
                rectangle = location[2]
                pt1, pt2 = rectangle.points()
                cv2.rectangle(overlay, pt1, pt2, color=150, thickness=-1)
            else:
                radius = location[2]
                cv2.circle(overlay, area_position, radius=int(radius), color=150, thickness=-1)

        # Draw box outline
        cv2.line(output, tuple(self.df["Corner"]["UL"].values[0]), tuple(self.df["Corner"]["UR"].values[0]), color=250, thickness=2)
        cv2.line(output, tuple(self.df["Corner"]["UL"].values[0]), tuple(self.df["Corner"]["LL"].values[0]), color=250, thickness=2)
        cv2.line(output, tuple(self.df["Corner"]["LL"].values[0]), tuple(self.df["Corner"]["LR"].values[0]), color=250, thickness=2)
        cv2.line(output, tuple(self.df["Corner"]["LR"].values[0]), tuple(self.df["Corner"]["UR"].values[0]), color=250, thickness=2)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        self.template = output


    def __call__(self):
        pass



df = pd.read_csv('./data/ehmt1/VideoNamesStatus.csv')
df_boxloc = pd.read_csv('./data/ehmt1/BoxLocations.csv', header=[0, 1, 2])

myvid = df["VideoName"][450]
temp = BoxTemplate(myvid)

temp.df

temp._set_locs()

#
# blank_image = np.zeros((300, 300), np.uint8)
# overlay = blank_image.copy()
# output = blank_image.copy()
#
# alpha = 0.2
# cv2.circle(overlay, (150,150), radius=30, color=150 , thickness=-1)
# cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
#
# # show the output image
# print("alpha={}, beta={}".format(alpha, 1 - alpha))
# cv2.imshow("Output", output)
# cv2.waitKey(0)
temp.template()


cv2.imshow('Templateee', temp.template)
cv2.waitKey(0)
cv2.destroyAllWindows()