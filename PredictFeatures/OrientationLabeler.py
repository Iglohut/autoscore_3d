import pandas as pd
import skvideo.io
import numpy as np
import cv2
import math
import skvideo.io

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


def _showframe_arrow(df, config, video_name, from_frame = 0):
    cap = cv2.VideoCapture(str(video_name))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r = 0.04 * height
    cap.set(1, from_frame)
    for i in range(from_frame , from_frame + 999999):
        ret, frame = cap.read()
        if not ret:
            print("Grab frame unsuccessful. ABORT MISSION!")
            break

        for _, cat in enumerate(config):
            frame = draw_arrow(frame, df, i, config, cat, r)

        cv2.imshow('frame: ' + str(i), frame)
        # Set waitKey
        key = cv2.waitKey()
        if key == ord('q'):
            break
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

def make_angles(df, ref, point):
    x1 = df[list(zip(point,["x"] * len(point)))].mean(axis=1)
    y1 = df[list(zip(point,["y"] * len(point)))].mean(axis=1)

    x2 = df[list(zip(ref,["x"] * len(ref)))].mean(axis=1)
    y2 = df[list(zip(ref,["y"] * len(ref)))].mean(axis=1)
    dx = x1 - x2
    dy = y2 - y1  # Because pixels are weirdly reversed for y: bottom is high
    angles = [math.atan2(dy[i],dx[i]) for i in range(len(dx))] # math.atan2 takes scalars
    return angles


def likelihood_point(df, point):
    meanlike = df[list(zip(point, ["likelihood"] * len(point)))].mean(axis=1)
    return meanlike


def draw_arrow(frame, df, i , config, cat, r):
    drawpoint = config[cat]["draw"]

    # Get most likely angle
    angles = [(df[(cat, point + "_likelihood")][i], df[(cat, point)][i], point) for point in config[cat]["points"]]
    best_angle = max(angles)

    # Draw
    width, height, _ = np.shape(frame)
    idx = list(config.keys()).index(cat)
    for j, angletext in enumerate(angles):
        # params
        angle = angletext[1]
        if angletext[2] == best_angle[2]: # Make chosen angle salient
            colour = config[cat]["colour"][angletext[2]]
            thickness = 2
        else:
            colour = (119, 136, 153)
            thickness = 1

        # Draw text
        y = int((idx * 0.2 + 0.1 + j * 0.05) * height)
        x = int(width * 0.6)
        text = cat + ", " + str(angletext[2]) + ": " + str(np.round(angletext[0], 6))
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thickness=2)

        # Draw arrow
        x1 = df[drawpoint].loc[i, (slice(None), 'x')].mean()
        y1 = df[drawpoint].loc[i, (slice(None), 'y')].mean()
        pt1 = np.asarray([x1, y1], dtype=np.int) # Point arrow rom this pixel
        direction = r * np.asarray([math.cos(angle), -math.sin(angle)])
        pt2 = pt1.astype(int) + direction.astype(int)
        cv2.arrowedLine(frame, tuple(pt1), tuple(pt2), colour, thickness=thickness)
    return frame



# Set paths
vp = '/home/iglohut/OneDrive/RU/Double Internship/Code/MatteoVideos/cropped_m62032-10182018145133-0000.mp4'
csvp ='/home/iglohut/OneDrive/RU/Double Internship/Code/MatteoVideos/cropped_m62032-10182018145133-0000_WITH-DLC-trained-on-cropped-videos.csv'
csvp_save = '/home/iglohut/OneDrive/RU/Double Internship/Code/MatteoVideos/cropped_m62032-10182018145133-0000_WITH-DLC-trained-on-cropped-videos-ANGLES.csv'
output_path = '/home/iglohut/OneDrive/RU/Double Internship/Code/MatteoVideos/cropped_m62032-10182018145133-0000-ANGLES.mp4'

# Open DataFrame
df = pd.read_csv(csvp, header=[0, 1], skipinitialspace=True, skiprows=1) # If scorer is in add ", skiprows=1"


config = {"angle1": {"ref": ["I_Back_drive", "I_Left_drive", "I_Right_drive"],
                     "points": {'front': ["I_Front_drive"],
                                'alwaysnose': ["I_nose_always"],
                                "visnose": ["I_nose_visible"]},

                     "draw": ["I_Back_drive", "I_Left_drive", "I_Right_drive"], # Draw is the point where to draw the arrow from
                     "colour": {"front":(0, 0, 255),
                                "alwaysnose": (0, 255, 0),
                                "visnose": (0, 255, 255)}},

           "angle2": {"ref": ["NI_leftear", "NI_rightear", "NI_BC"],
                      "points": {'point1': ["NI_nose"]},

                      "draw": ["NI_leftear", "NI_rightear", "NI_BC"],

                      "colour": {"point1": (255, 0, 0)}}}


# Here we write the angles into the dataframe
for i, cat in enumerate(config):
    for point_ in config[cat]["points"]:
        ref = config[cat]["ref"]
        # point = config[cat]["point"]
        point = config[cat]["points"][point_]
        df[(cat, point_)] = make_angles(df,ref, point)
        df[(cat, point_ + "_likelihood")] = likelihood_point(df, point)
df.to_csv(csvp_save)

# Here we write the video
cap = cv2.VideoCapture(str(vp))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
r = 0.04 * height
writer = skvideo.io.FFmpegWriter(output_path)

for i in range(_vidlength(vp)):
    ret, frame = cap.read()
    if not ret:
        print("Grab frame unsuccessful. ABORT MISSION!")
        break

    for _, cat in enumerate(config):
        frame = draw_arrow(frame, df, i , config, cat, r)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    writer.writeFrame(frame)
cap.release()
writer.close()