import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv('./data/ehmt1/VideoNamesStatus.csv')
df_boxloc = pd.read_csv('./data/ehmt1/BoxLocations.csv')

# TODO Visalize box outline + object/corner radius template ON ANY VID OF ROUND

class BoxTemplate:
    df_boxloc = pd.read_csv('./data/ehmt1/BoxLocations.csv')
    def __init__(self, video_path):
        #TODO get vid round.. maybe select random frame later for illustration purposes