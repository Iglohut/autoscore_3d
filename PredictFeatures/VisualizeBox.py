import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import PredictFeatures.ShapeHelper as PS
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

    # >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
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
        self.trial = int(video_path.split("_t0")[-1].split("_")[0])  # Trial number - because the camera flip...
        self._grab_midframe() # Get a frame grayscale
        self._videodimension() # Set dimension of box
        self._boxcolor()

        # Select correct template
        self.df = self.df_boxloc.loc[self.df_boxloc["Round"]["Round"]["Round"] == self.round] # Select BoxLocations of this round
        self.df = self.df[self.df["BoxColor"]["BoxColor"]["BoxColor"] == self.boxcolor]
        if self.round == 7 and self.trial >= 21:
            self.df = self.df[self.df["Trial_flip"]["Trial_flip"]["Trial"] == 21]
        if self.round == 7 and self.trial < 21:
            self.df = self.df[self.df["Trial_flip"]["Trial_flip"]["Trial"] == 0]

        self._correct_shapes_overengineered() # Correct for camera changes scale.. :\

        self._set_locs() # Set per location the radius of detection --> self.full_locations

    def _correct_shapes_overengineered(self):
        """Re-scale the location positions according to video. This is because scorer32 allows for rescale-saving and thus inconsistent shape of videos."""
        if self.round in [8, 9] and self.width != 640:  # Round 8/9 meta: because scorer32 allows scaling
            locations = self.df.columns.values[4::].tolist()  # Select columns object/corner/wall and sublocations to iterate over
            for location in locations:  # For every possible mapped location
                self.df.loc[:, location] = int(self.df.loc[:, location].values[0] * (self.width / 640)) # rescale locations


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
        not_green = clt[::len(clt)-1]
        if clt[1] > np.mean(not_green) + np.std(not_green) and np.std(not_green) > 20:
            color = "Green"
        else:
            color = "White"
        self.boxcolor = color
        cap.release()

    def _rotate_result(self, closest_area):
        """

        :param degrees: degrees as quadrant moves: [0 , 1,  2,  3] = [0, 90, 180, 270]
        :return:
        """
        # pass
        walls = ["North", "East", "South", "West"]
        corners = ["UR", "LR", "LL", "UL"]
        indices = [0, 1, 2, 3] * 3
        degrees = int(self.df["Trial_flip"]["Trial_flip"]["Degrees"].values[0])

        distance = closest_area[0]
        superlocation = closest_area[1]
        sublocation = closest_area[2]

        if superlocation == "Wall":
            idx = walls.index(sublocation)
            idx += degrees # new idx
            new_sublocation = walls[indices[idx]]
        elif superlocation in ["Corner", "Object"]:
            idx = corners.index(sublocation)
            idx += degrees # new idx
            new_sublocation = corners[indices[idx]]

        return (distance, superlocation, new_sublocation)

    def _set_locs(self):
        """Sets, for pivot each location, the paramaters for its shape where something is detected.
        self.full_locations"""
        locations = self.df.columns.values[4::2].tolist() # Select columns object/corner/wall and sublocations to iterate over
        self.full_locations = []
        for location in locations: # For every possible mapped location
            superlocation = location[0] # Object/Corner/Wall
            sublocation = location[1] # Left/West etc
            area_position = self.df[superlocation][sublocation]  # x,y coordinate

            if superlocation == "Object":
                radius = int(np.linalg.norm(area_position - self.df["Corner"][sublocation]) / 2)
            elif superlocation == "Corner":
                radius = int(np.linalg.norm(area_position - self.df["Object"][sublocation]) / 2.9)
            elif superlocation == "Wall": # Wall are represented as rectangles!
                wall_length = 12 # TODO Make wall full length and overlap with corner
                wall_width = 4
                sidewall_extension = 300
                if sublocation == "North":
                    dx =abs(self.df["Corner"]["UR"]['x'].values[0] - self.df["Corner"]["UL"]['x'].values[0])

                    x1 = self.df["Corner"]["UL"]['x'].values[0] + dx / wall_length
                    x2 = self.df["Corner"]["UR"]['x'].values[0] - dx / wall_length

                    dy = abs((self.df["Corner"]["UR"]['y'].values[0] - self.df["Object"]["UL"]['y'].values[0]))
                    y1 = self.df["Corner"]["UR"]['y'].values[0] + dy / wall_width
                    y2 = self.df["Corner"]["UR"]['y'].values[0] - sidewall_extension * dy / wall_width

                elif sublocation == "South":
                    dx = abs(self.df["Corner"]["LR"]['x'].values[0] - self.df["Corner"]["LL"]['x'].values[0])

                    x1 = self.df["Corner"]["LL"]['x'].values[0] + dx / wall_length
                    x2 = self.df["Corner"]["LR"]['x'].values[0] - dx / wall_length

                    dy = abs((self.df["Corner"]["LR"]['y'].values[0] - self.df["Object"]["LL"]['y'].values[0]))
                    y1 = self.df["Corner"]["LR"]['y'].values[0] + sidewall_extension * dy / wall_width
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
        distances = []
        for location in self.full_locations:
            superlocation = location[0]
            sublocation = location[1]
            area_position = self.df[superlocation][sublocation].values[0]  # x,y coordinates

            if superlocation == "Wall": # Check if in rectangle
                rectangle = location[2]
                point = PS.Point(position[0], position[1])

                distance = int(not rectangle.contains(point)) * self.width/2 # Jus tto make it arbitrary positive if not in rectangle

            else: # Check distance from circles
                radius = location[2]
                distance = abs(np.linalg.norm(area_position - position)) - radius # L2 norm distance

            distances.append(self._rotate_result((distance, superlocation, sublocation)))

        closest_area = min(distances)

        return closest_area


    def detect(self, position):
        """
        :param position: tuple (x, y) of pixel position of animal
        :return: the pivot point in which the animal is (if any)
        """
        closest_pivot = self.closest(position)
        distance = closest_pivot[0]
        object_detection_error = 25 # in normalized pixels
        if distance <= 0:
            return closest_pivot[1:]
        elif closest_pivot[1] == "Object" and distance <= object_detection_error:
            return closest_pivot[1:]
        else:
            return None

    def template(self):
        "Image template of the box"
        frame = self.midframe
        # frame = np.zeros((self.height, self.width)) # Because cv2 transposes it
        overlay = frame.copy()
        output = frame.copy()
        alpha = 0.8

        for location in self.full_locations:
            superlocation = location[0]
            sublocation = location[1]
            area_position = tuple(self.df[superlocation][sublocation].values[0])  # x,y coordinates

            if superlocation == "Wall": # Draw wall rectangles
                rectangle = location[2]
                pt1, pt2 = rectangle.points()
                cv2.rectangle(overlay, pt1, pt2, color=150, thickness=-1)
            else: # Draw point circles
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

    def __len__(self):
        return len(self.full_locations)


# # # TODO increase wall length, not thickness ;;; corner smaller?
# df = pd.read_csv('./data/ehmt1/VideoNamesStatus.csv')
# # df_boxloc = pd.read_csv('./data/ehmt1/BoxLocations.csv', header=[0, 1, 2])
#
# myvid = df["VideoName"][2700]
# temp = BoxTemplate(myvid)
#
# temp.df
#
# temp._set_locs()
# temp.template()
#
#
# cv2.imshow('Templateee', temp.template)
# cv2.imwrite('/media/iglohut/Iglohut/BoxTemplate_example.png',temp.template)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # temp.closest([300,300])
#
# temp.closest([154, 254])


