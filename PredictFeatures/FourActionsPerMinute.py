import pandas as pd
import csv
import os
import json
import numpy as np

def split_list(l, n):
    """
    :param l: (nested) list
    :param n: n chunks to be split by
    :return: list of size n
    """
    d, r = divmod(len(l), n)
    mylist = []
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        # yield l[si:si + (d + 1 if i < r else d)]
        mylist.append(l[si:si + (d + 1 if i < r else d)])
    return mylist

def specific_sequence(actions, goal):
    """
    :param actions: (nested) list of actions
    :param goal: list of goal actions. e.g: ['obj_1', 'obj_1']
    :return: list of sequence containing only those goal actions
    """
    sequence = []
    for el in actions:
        stopper = None
        for subel in el:
            if stopper: # No duplicate actions of 1 frame
                pass
            elif subel in goal:
                sequence.append(subel)
                stopper=True
     return sequence


def total_action_chunks(actions, goal):
    """Returns the total number of continued bouts of one actions"""
    chunk_sequence =[]
    chunk = False
    for el in actions:
        stopper = None
        for subel in el:
            if stopper: # No duplicate actions of 1 frame
                pass
            elif subel in goal and not chunk:
                chunk_sequence.append(subel)
                stopper = True
                chunk = True
            elif subel not in goal:
                chunk = False
     return chunk_sequence


class DataSet:
    path_overview = os.getcwd() + '/data/ehmt1/VideoNamesStatus.csv'
    path_actions = os.getcwd() + '/data/ehmt1/ehmt1_actions'
    path_features = os.getcwd() + '/data/ehmt1/ehmt1_features_four'

    skip_percent_frames = 3 # Frames to skip at the beginning of video because it is random

    def __init__(self):
        self.df_overview = self.load_overview

    @property
    def load_overview(self):
        """
        Loads dataframe with all videonames and trial information.
        """
        df = pd.read_csv(self.path_overview)
        return df

    def load_actions(self, videoname):
        """
        :param videoname: name of video
        :return: action list with that video for all frames if it exists. Else it retrurns False.
        """
        files = os.listdir(self.path_actions)
        myfile = [file for file in files if videoname.split('/')[-1].split('.')[0] in file if '#' not in file]

        if len(myfile) > 0: # If file exists
            myfile = self.path_actions + "/" + myfile[0]
            with open(myfile, 'r') as f:
                reader = csv.reader(f)
                mylist = list(reader)
        else: # If file doesn't exist
            mylist = False
        return mylist


    def save_features_in_df(self, i, featuredict):
        """
        :param i: row number
        :param features: dictionary of features and it value
        :return:
        """

        # Load row as single row
        index = list(range(0, len(self)))
        index.remove(i)
        df = self.df_overview.drop(self.df_overview.index[index]) # Select row; or remove all other rows

        # Add features in df
        for feature in list(featuredict.keys()):
            df[feature] = featuredict[feature]

        # Save it
        path_df = self.path_features + '/'+ df.VideoName.unique()[0].split('.avi')[0].split('/')[-1] + '.csv'
        df.to_csv(path_df, index=False)
        print("Feature file saved to:", path_df)


    def __call__(self, i):
        """
        :param i: row number
        :return: trial info df and action list of first 5 minutes f any trial
        """
        if i > len(self) or np.isnan(self.df_overview.subject[i]) or self.df_overview.StatusPredicted[i] < 3: # If row cannot exist
            return False
        # Load actions corresponding video
        actions = self.load_actions(self.df_overview.VideoName[i])

        # Cut first x frames: last SHOULD be correct
        actions = actions[int(len(actions) * (self.skip_percent_frames / 100)):]


        # If trial 21, cut to 5 minutes: half the list
        if self.df_overview.sequence_nr[i]  == 21:
            actions = actions[:len(actions)//2]

        return actions


    def __len__(self):
        """Length is length of all the rows in overview."""
        return len(self.df_overview)



class AnimalFeatures:
    # features = [
    #             "min1_explore_time", "min2_explore_time", "min3_explore_time", "min4_explore_time", "min5_explore_time",
    #             "min1_wall_time", "min2_wall_time", "min3_wall_time", "min4_wall_time", "min5_wall_time",
    #             "min1_corner_time", "min2_corner_time", "min3_corner_time", "min4_corner_time", "min5_corner_time",
    #             "min1_other_time", "min2_other_time", "min3_other_time", "min4_other_time", "min5_other_time"
    #             ] # More to be added if complete this successfully
    features = [
                "min5_explore_time",
                "min5_wall_time",
                "min5_corner_time",
                "min5_other_time"
                ]
    timescale = 300 # 300 seconds. everything is normalised, but i can times this

    def __init__(self, actions):
        """
        :param actions: (nested) list of actions: Corner/Wall/obj_1/obj_2/''
        Assumes that the list is about a 5 minute event!
        """
        self.actions = actions # Store actions
        self.actions_split = split_list(self.actions, 5) # Split into 5 parts; 1 per minute!
        self.featuredict = dict.fromkeys(self.features) # Make dictionary of feature

    def __repr__(self):
        """Makes calling the class showing the features in nice way"""
        return json.dumps(self.featuredict, indent=4, sort_keys=True)

    def get_actions_upuntil(self, minute=5):
        """
        Get all actions up until that minute; so not split by but full list.
        Is to make cumulative expressions more efficient.
        """
        actions = []
        for idx in range(minute):
            actions += self.actions_split[idx]
        return actions

    def calc_actiontime(self, minute, actions):
        """
            Calculate relative to all frames exploretime
        :param minute: up until minute
        :param action: for the action in list [action_i]
        :return: n_acion/total_n
        """
        action_sequence = specific_sequence(self.get_actions_upuntil(minute), goal=actions)
        actions_n = len(action_sequence)
        return self.timescale * (actions_n / len(self.actions))

    def make_features(self):
        # Exploretimes
        # self.featuredict['min1_explore_time'] = self.calc_actiontime(1, ['obj_1', 'obj_2'])
        # self.featuredict['min2_explore_time'] = self.calc_actiontime(2, ['obj_1', 'obj_2'])
        # self.featuredict['min3_explore_time'] = self.calc_actiontime(3, ['obj_1', 'obj_2'])
        # self.featuredict['min4_explore_time'] = self.calc_actiontime(4, ['obj_1', 'obj_2'])
        self.featuredict['min5_explore_time'] = self.calc_actiontime(5, ['obj_1', 'obj_2'])

        # Wall times
        # self.featuredict['min1_wall_time'] = self.calc_actiontime(1, ['Wall'])
        # self.featuredict['min2_wall_time'] = self.calc_actiontime(2, ['Wall'])
        # self.featuredict['min3_wall_time'] = self.calc_actiontime(3, ['Wall'])
        # self.featuredict['min4_wall_time'] = self.calc_actiontime(4, ['Wall'])
        self.featuredict['min5_wall_time'] = self.calc_actiontime(5, ['Wall'])

        # Corner times
        # self.featuredict['min1_corner_time'] = self.calc_actiontime(1, ['Corner'])
        # self.featuredict['min2_corner_time'] = self.calc_actiontime(2, ['Corner'])
        # self.featuredict['min3_corner_time'] = self.calc_actiontime(3, ['Corner'])
        # self.featuredict['min4_corner_time'] = self.calc_actiontime(4, ['Corner'])
        self.featuredict['min5_corner_time'] = self.calc_actiontime(5, ['Corner'])


        # Other times
        # self.featuredict['min1_other_time'] = 60 - self.featuredict['min1_explore_time'] - self.featuredict['min1_wall_time'] - self.featuredict['min1_corner_time']
        # self.featuredict['min2_other_time'] = 120 - self.featuredict['min2_explore_time'] - self.featuredict['min2_wall_time'] - self.featuredict['min2_corner_time']
        # self.featuredict['min3_other_time'] = 180 - self.featuredict['min3_explore_time'] - self.featuredict['min3_wall_time'] - self.featuredict['min3_corner_time']
        # self.featuredict['min4_other_time'] = 270 - self.featuredict['min4_explore_time'] - self.featuredict['min4_wall_time'] - self.featuredict['min4_corner_time']
        self.featuredict['min5_other_time'] = self.calc_actiontime(5, [''])

# CALCULATE FEATURES
# data = DataSet()
# for i in range(len(data)):
#     actions = data(i)
#     if actions:
#         animal = AnimalFeatures(actions)
#         animal.make_features()
#         data.save_features_in_df(i, animal.featuredict)
#
#
#
# # SAVE ALL DFS AS ONE IN DATA
# dfs =[]
# files = os.listdir(DataSet.path_features)
# for file in files:
#     df = pd.read_csv(DataSet.path_features +'/' + file)
#     dfs.append(df)
#
# superdf = pd.concat(dfs)
# superdf.to_csv(os.getcwd() + '/data/ehmt1/SS_alldata_autoscored_fouractions.csv', index=False)




# DATA ANALYSIS STUFF

df = pd.read_csv(os.getcwd() + '/data/ehmt1/SS_alldata_autoscored_fouractions.csv')

# df.groupby('genotype').agg(['mean','sem']).drop(('genotype','sem'), axis=1)


# Make empty dataframe
index = pd.MultiIndex.from_product([['WT', 'KO']],
                                   names=['genotype'])
columns = pd.MultiIndex.from_product([AnimalFeatures.features, ['Mean', 'SEM']],
                                     names=['Feature', 'Statistics'])
df_stats = pd.DataFrame([], index=index, columns=columns)


# Fill in blanks
genotypes = ['WT', 'KO']

for igen, genotype in enumerate(genotypes):
    for feature in AnimalFeatures.features:
        SEM = df[df['genotype'] == igen].sem()[feature]
        MEAN = df[df['genotype'] == igen].mean()[feature]
        df_stats.loc[(genotype), (feature, 'SEM')] = np.copy(SEM)
        df_stats.loc[(genotype), (feature, 'Mean')] = np.copy(MEAN)


df_stats.to_csv(os.getcwd() + '/data/ehmt1/SS_alldata_autoscored_fouractions_STATS.csv', index=False)

