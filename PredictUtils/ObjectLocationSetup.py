import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2


class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def plot(frame):
    # data = np.random.random((300,300))
    fig, ax = plt.subplots()
    im = ax.imshow(frame, cmap='gray', interpolation='none')
    ax.format_coord = Formatter(im)
    plt.show()


def showframe(video_name):
    cap = cv2.VideoCapture(str(video_name))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(1, int(frameCount/2))
    ret, frame = cap.read()
    GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:
        plot(GRAY)
    else:
        print("\nI failed loading the frame")




video_name = '/media/iglohut/MD_Smits/Internship/Intellectual_Disability/Intellectual_Disability/round_9/mouse_training_OS_5trials_inteldis_59_66_or_206_13/mouse_training_OS_5trials_inteldis_59_66_or_206_13_t0003_raw.avi'
showframe(video_name)

# df = pd.read_csv('/media/iglohut/MD_Smits/Internship/autoscore_3d/PredictUtils/BoxLocations.csv', header=[0, 1, 2])