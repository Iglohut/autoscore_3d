import cv2
import math
import numpy as np
import skvideo.io
import os

class IconFrame():
    wall_path = os.getcwd() + "/PredictFeatures/icons/trumpwall.png"
    corner_path = os.getcwd() + "/PredictFeatures/icons/corner.png"
    explore_path = os.getcwd() + "/PredictFeatures/icons/explore.png"
    def __init__(self, frame):
        self.frame = self._load_frame(frame)
        self.new_frame = self.frame.copy()
        self.wall = cv2.resize(cv2.imread(self.wall_path, -1), self.goalsize)
        self.corner = cv2.resize(cv2.imread(self.corner_path, -1), self.goalsize)
        self.explore = cv2.resize(cv2.imread(self.explore_path, -1), self.goalsize)

    @property
    def goalsize(self):
        sizereduction = 7
        dimensions = np.array(np.array(self.frame.shape) / sizereduction).astype(np.int)
        return (dimensions[1], dimensions[0])


    def _load_frame(self, frame):
        if len(frame.shape) == 3:
            return frame
        else:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    def embed_icon(self, overlay, x_offset, y_offset, chosen = 0):
        """Embed single icon on frame"""
        y1, y2 = y_offset, y_offset + overlay.shape[0]
        x1, x2 = x_offset, x_offset + overlay.shape[1]
        if chosen == 0: alpha = 700
        if chosen == 1:
            alpha =255
            # Change color of icon
            old_alpha = overlay[:, :, 3].copy()
            if (overlay == self.wall).all(): # Change color to orange
                overlay[(np.where(overlay[:, :, 3] > 0))] = [0, 33, 166, 255]
            elif (overlay == self.corner).all():
                overlay[(np.where(overlay[:, :, 3] > 0))] = [0, 190, 0, 255]
            elif (overlay == self.explore).all():
                overlay[(np.where(overlay[:, :, 3] > 0))] = [205, 0, 0, 255]
            overlay[:, :, 3] = old_alpha

        alpha_s = overlay[:, :, 3] / alpha
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            self.new_frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                      alpha_l * self.new_frame[y1:y2, x1:x2, c])



    def embed_icons(self, actions = None):
        """embeds all icons on frame, but only makes the one that has an action salient."""
        dimensions = np.array(np.array(self.frame.shape))[0:2]

        if type(actions) is not list: actions = [actions]

        chosen_wall = 0
        chosen_object = 0
        chosen_corner = 0
        for action in actions:
            if action in ["Wall", 'wall']:
                chosen_wall = 1
            if action in ["Object", "obj_1", "obj_2"]:
                chosen_object = 1
            if action in ["Corner"]:
                chosen_corner = 1

        y_loc = int(dimensions[0] * 0.01)
        self.embed_icon(self.wall, int(dimensions[1] * 0.3), y_loc, chosen_wall)
        self.embed_icon(self.corner, int(dimensions[1] * 0.15), y_loc, chosen_corner)
        self.embed_icon(self.explore, int(dimensions[1] * 0.01), y_loc, chosen_object)


    def __call__(self):
        return self.new_frame

def draw_arrow(frame, df, i , config, cat, r):
    drawpoint = config[cat]["draw"]

    # Get most likely angle
    angles = [(df[(cat, point + "_likelihood")][i], df[(cat, point)][i], point) for point in config[cat]["points"]]
    best_angle = max(angles)


    # Draw
    width, height, _ = np.shape(frame)
    idx = list(config.keys()).index(cat)
    ranked_chosen = False
    for j, angletext in enumerate(angles):
        # params
        angle = angletext[1]
        if angletext[0] >= 0.8 and not ranked_chosen: # This makes it so
            colour = config[cat]["colour"][angletext[2]]
            thickness = 2
            ranked_chosen = True
        elif angletext[2] == best_angle[2] and not ranked_chosen: # Make chosen angle salient
            colour = config[cat]["colour"][angletext[2]]
            thickness = 2
            ranked_chosen = True
        else:
            colour_ = config[cat]["colour"][angletext[2]]
            colour = tuple([int(0.25 * x + 0.5 * max(colour_)) for x in colour_]) # Colour saturation
            thickness = 1
        # Draw arrow
        x1 = df[drawpoint].loc[i, (slice(None), 'x')].mean()
        y1 = df[drawpoint].loc[i, (slice(None), 'y')].mean()
        pt1 = np.asarray([x1, y1], dtype=np.int) # Point arrow rom this pixel
        direction = r * np.asarray([math.cos(angle), -math.sin(angle)])
        pt2 = pt1.astype(int) + direction.astype(int)
        cv2.arrowedLine(frame, tuple(pt1), tuple(pt2), colour, thickness=thickness)
    return frame


def draw_limbs(frame, df, i, config):
    bodylimbs = df.columns.tolist()[1:]
    bodylimbs = [limb for limb in bodylimbs if ("likelihood" not in limb) and (any(s not in limb for s in list(config.keys())))]
    colours = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
               [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
               [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

    width, height, _ = np.shape(frame)
    colour = 0
    for idx in range(0, len(bodylimbs), 2):
        bodylimb_x = int(df[bodylimbs[idx]][i])
        bodylimb_y = int((df[bodylimbs[idx + 1]][i]))
        cv2.circle(frame, (bodylimb_x, bodylimb_y), radius=2, color=colours[colour], thickness=-1)
        colour += 1
    return frame

def _vidlength(video_name):
    """
    :param video_name: path to video
    :return: number of frames in video
    """
    cap = cv2.VideoCapture(str(video_name))
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    length = int(cv2.VideoCapture.get(cap, property_id))
    return length


def make_video(SequenceObject):
    """
    :param vp: videopath
    :param df: df_pose with HD
    :param actions: action sequence... ["Wall", None, "Corner", "Explore,..., None]
    :return:
    """

    vp = SequenceObject.template.vidname
    df = SequenceObject.df_pose
    actions = SequenceObject.ActionSequence["actions"]
    # vp = '/media/iglohut/MD_Smits/Internship/Intellectual_Disability/Intellectual_Disability/round_8/mouse_training_OS_5trials_inteldis_45_53_or_110_18/mouse_training_OS_5trials_inteldis_45_53_or_110_18_t0002_raw.avi'
    #
    output_path = vp.split('.')[0] + "_IGLOHUT." + 'mp4'
    print("Going to save video at:", output_path)

    cap = cv2.VideoCapture(str(vp))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    r = 0.04 * height
    writer = skvideo.io.FFmpegWriter(output_path)

    for i in range(_vidlength(vp)):
        ret, frame = cap.read()
        #
        # if i in range(5848, 6100):
        if not ret:
            print("Grab frame unsuccessful. ABORT MISSION!")
            break

        for _, cat in enumerate(config):
            frame = draw_arrow(frame, df, i , config, cat, r)
            frame = draw_limbs(frame, df, i, config)


        myframe = IconFrame(frame)
        myframe.embed_icons(actions[i])


        # # text debug headDirection
        # y = int(height / 2)
        # x = int(0)
        # text =str("HeadDirection: {:.4}".format(SequenceObject.headDirection(i)))
        # cv2.putText(myframe(), text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)

        # Print frame nr
        y = int(height/15)
        x = int(width * 0.8)
        # cv2.putText(myframe(), str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), thickness=1)




        frame = cv2.cvtColor(myframe(), cv2.COLOR_BGR2RGB)
        writer.writeFrame(frame)
    cap.release()
    writer.close()


config = {"angle1": {"ref": ["Right  ear", "Left ear", "Back"],
                     "points": {'Nose': ["Nose"]},
                     "draw": ["Right  ear", "Left ear", "Back"],
                     "colour": {'Nose': (71, 99, 255)}}}



# frame = cv2.imread('/media/iglohut/Iglohut/BoxTemplate_example1.png')
# myframe = IconFrame(frame)
# myframe.embed_icons(["Wall"])
#
# cv2.imshow('icons', myframe())
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
