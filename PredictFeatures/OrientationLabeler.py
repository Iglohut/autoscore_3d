import pandas as pd
import skvideo.io
import cv2
import math
from pykalman import KalmanFilter
import numpy as np
import os

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
            frame = draw_limbs(frame, df, i, config)

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

        # # Draw text
        # y = int((idx * 0.2 + 0.1 + j * 0.05) * height)
        # x = int(width * 0.6)
        # text = cat + ", " + str(angletext[2]) + ": " + str(np.round(angletext[0], 6))
        # cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thickness=2)

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
        #
        # if bodylimb_x > width:
        #     bodylimb_x = width
        # elif bodylimb_x < 0:
        #     bodylimb_x = int(0)
        # if bodylimb_y > height:
        #     bodylimb_y = height
        # elif bodylimb_y < 0:
        #     bodylimb_y = int(0)

        cv2.circle(frame, (bodylimb_x, bodylimb_y), radius=2, color=colours[colour], thickness=-1)
        colour += 1
    return frame


def myKalman(measurements):
    """
    This Kalman smoothes a timeseries of (x, y) coordinates for a single limb.
    :param measurements: coordinate tuple list [(x1, y1), ..., (xn, yx)]
    :return: kalman smoothed coordinates list: [[x]], [y]]
    """
    # Set initial assumptions
    initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0]
    # transition_matrix = [[1, 1, 0, 0],
    #                      [0, 1, 0, 0],
    #                      [0, 0, 1, 1],
    #                      [0, 0, 0, 1]]

    transition_matrix = [[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]


    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean)


    kf1 = kf1.em(measurements, n_iter=5)

    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)
    # # Kalman2 based on kalman1 observed covariance + more uncertainty for smoothing
    # kf2 = KalmanFilter(transition_matrices = transition_matrix,
    #                   observation_matrices = observation_matrix,
    #                   initial_state_mean = initial_state_mean,
    #                   observation_covariance = 5*kf1.observation_covariance,
    #                   em_vars=['transition_covariance', 'initial_state_covariance'])
    #
    #
    # kf2 = kf2.em(measurements, n_iter=5)
    # (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)
    return [smoothed_state_means[:, 0], smoothed_state_means[:, 2]]

def kalman_df(df):
    """
    Doe skalman smoothing on all bodyparts of df
    :param df: DataFrame with only bodyparts (no HD yet!)
    :return: smoothed df
    """
    cols = df.columns.values.tolist()[1:]
    cols = [col for col in cols if "likelihood" not in col]

    for idx in range(0, len(cols), 2):
        bodylimb_x = cols[idx]
        bodylimb_y = cols[idx + 1]
        measurements = np.asarray(list(zip(list(df[bodylimb_x]), list(df[bodylimb_y]))))

        estimated_measurements = myKalman(measurements)

        estimated_x = estimated_measurements[0]
        estimated_y = estimated_measurements[1]

        df[bodylimb_x] = estimated_x
        df[bodylimb_y] = estimated_y
    return df




posefolder = '/media/iglohut/MD_Smits/Internship/autoscore_3d/data/ehmt1/ehmt1_poses'


vp = '/media/iglohut/MD_Smits/Internship/Intellectual_Disability/Intellectual_Disability/round_8/mouse_training_OS_5trials_inteldis_45_53_or_110_18/mouse_training_OS_5trials_inteldis_45_53_or_110_18_t0002_raw.avi'
csvp ='/media/iglohut/MD_Smits/Internship/Intellectual_Disability/Intellectual_Disability/round_8/mouse_training_OS_5trials_inteldis_45_53_or_110_18/mouse_training_OS_5trials_inteldis_45_53_or_110_18_t0002_rawDeepCut_resnet50_OS_poseDec10shuffle1_1030000.csv'

csvp_save = csvp.split('.csv')[0] + "_ORI.csv"
output_path = vp.split('.')[0] + "_ORI." + vp.split('.')[1]

# Open DataFrame
df = pd.read_csv(csvp, header=[0, 1], skipinitialspace=True, skiprows=1) # If scorer is in add ", skiprows=1"


config = {"angle1": {"ref": ["Right  ear", "Left ear", "Back"],
                     "points": {'Nose': ["Nose"]},
                     "draw": ["Right  ear", "Left ear", "Back"],
                     "colour": {'Nose': (71, 99, 255)}}}



# df = kalman_df(df) # Does Kalman filter on original DLC csv

# Here we write the angles into the dataframe
for i, cat in enumerate(config):
    for point_ in config[cat]["points"]:
        ref = config[cat]["ref"]
        # point = config[cat]["point"]
        point = config[cat]["points"][point_]
        df[(cat, point_)] = make_angles(df,ref, point)
        df[(cat, point_ + "_likelihood")] = likelihood_point(df, point)
# df.to_csv(csvp_save, index=False)


# _showframe_arrow(df, config, vp, from_frame = 500)

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
        frame = draw_limbs(frame, df, i, config)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    writer.writeFrame(frame)
cap.release()
writer.close()
