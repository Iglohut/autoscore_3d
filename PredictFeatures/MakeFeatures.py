import pandas as pd
import csv
import os
import json
import numpy as np
import pysal

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
    path_features = os.getcwd() + '/data/ehmt1/ehmt1_features'

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
    features = ["first_object", "first_object_latency", "stay1", "stay2", "SS1", "perseverance", "n_transitions",
                "min1_n_explore", "min2_n_explore", "min3_n_explore", "min4_n_explore", "min5_n_explore",
                'min1_n_explore_obj1', 'min2_n_explore_obj1', 'min3_n_explore_obj1', 'min4_n_explore_obj1', 'min5_n_explore_obj1',
                'min1_n_explore_obj2', 'min2_n_explore_obj2', 'min3_n_explore_obj2', 'min4_n_explore_obj2', 'min5_n_explore_obj2',
                "min1_obj1_time", "min2_obj1_time", "min3_obj1_time", "min4_obj1_time", "min5_obj1_time",
                "min1_obj2_time", "min2_obj2_time", "min3_obj2_time", "min4_obj2_time", "min5_obj2_time",
                "min1_DI", "min2_DI", "min3_DI", "min4_DI", "min5_DI",
                "min1_explore_time", "min2_explore_time", "min3_explore_time", "min4_explore_time", "min5_explore_time",
                "bout_time", "bout_obj1_time", "bout_obj2_time"] # More to be added if complete this successfully

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

    def calc_first_object(self):
        """First object explored"""
        object_sequence = specific_sequence(self.actions, goal=['obj_1', 'obj_2']) + [np.nan]
        return object_sequence[0]

    def calc_first_xaction_latency(self, actions):
        """
            Calculate the latency to first time doing action in actions
        :param actions: for the action in list [action_i]
        :return:
        """
        for i, actions_frame in enumerate(self.actions):
            for action in actions_frame:
                if action in actions:
                    return self.timescale * (i / len(self.actions))


    def calc_DI(self, minute=5):
        """
        Calculate cumulative DI up until that minute
        """
        object_sequence = specific_sequence(self.get_actions_upuntil(minute), goal=['obj_1', 'obj_2'])

        obj1_n = len([obj for obj in object_sequence if obj == 'obj_1'])
        obj2_n = len(object_sequence) - obj1_n

        try:
            DI = (obj2_n - obj1_n) / (obj2_n + obj1_n)
        except ZeroDivisionError:
            DI = 0
        return DI

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

    def calc_action_n(self, minute, actions):
        """
            Calculate total amount of bouts of one action
            This means if one action is done for consecutive frames, this counts as 1 continued action.
        :param minute: up until minute
        :param actions: for the action in list [action_i]
        :return: total n actions as bouts
        """
        actions_n = len(total_action_chunks(self.get_actions_upuntil(minute), goal=actions))
        return actions_n

    def calc_markov_params(self):
        """
            Calculate markov properties of object exploration:
        :return: transition stay at same objects, steadystate, and perseverance same object
        """

        object_names = ['obj_1', 'obj_2']
        object_sequence = total_action_chunks(self.actions, goal=object_names)
        m = pysal.Markov(np.array([object_sequence])) # MarkovModel

        # Iinitiate empty variables
        stay1 = np.nan
        stay2 = np.nan
        SS1 = np.nan  # Steady state object 1
        perseverance = np.nan  # Tendency to not move
        classes = m.classes.tolist()

        if set(object_names).issubset(classes):  # If mouse explored all objects
            index_object1 = classes.index(object_names[0])
            index_object2 = classes.index(object_names[1])
            stay1 = m.p[index_object1, index_object1]
            stay2 = m.p[index_object2, index_object2]
            SS1 = m.steady_state[index_object1, 0]
            perseverance = (m.p.diagonal().sum() - np.fliplr(m.p).diagonal().sum()) / m.p.sum()

        elif object_names[0] in classes:  # If object 1 and not object 2 in
            index_object = classes.index(object_names[0])
            stay1 = m.p[index_object, index_object]
            SS1 = 1  # Obviously if only when only to object 1 its 1
            perseverance = 1

        elif object_names[1] in classes:  # If object 2 and not object 1 in
            index_object = classes.index(object_names[1])
            stay1 = m.p[index_object, index_object]
            perseverance = 1
        return stay1, stay2, SS1, perseverance


    def make_features(self):

        # First object info
        self.featuredict['first_object'] = self.calc_first_object()
        self.featuredict['first_object_latency'] = self.calc_first_xaction_latency(['obj_1', 'obj_2'])

        # Markov info
        stay1, stay2, SS1, perseverance = self.calc_markov_params()
        self.featuredict['stay1'] = stay1
        self.featuredict['stay2'] = stay2
        self.featuredict['SS1'] = SS1
        self.featuredict['perseverance'] = perseverance
        self.featuredict['n_transitions'] = self.calc_action_n(5, ['obj_1', 'obj_2']) - 1

        # n_explores
        self.featuredict['min1_n_explore'] = self.calc_action_n(1, ['obj_1', 'obj_2'])
        self.featuredict['min2_n_explore'] = self.calc_action_n(2, ['obj_1', 'obj_2'])
        self.featuredict['min3_n_explore'] = self.calc_action_n(3, ['obj_1', 'obj_2'])
        self.featuredict['min4_n_explore'] = self.calc_action_n(4, ['obj_1', 'obj_2'])
        self.featuredict['min5_n_explore'] = self.calc_action_n(5, ['obj_1', 'obj_2'])

        self.featuredict['min1_n_explore_obj1'] = self.calc_action_n(1, ['obj_1'])
        self.featuredict['min2_n_explore_obj1'] = self.calc_action_n(2, ['obj_1'])
        self.featuredict['min3_n_explore_obj1'] = self.calc_action_n(3, ['obj_1'])
        self.featuredict['min4_n_explore_obj1'] = self.calc_action_n(4, ['obj_1'])
        self.featuredict['min5_n_explore_obj1'] = self.calc_action_n(5, ['obj_1'])

        self.featuredict['min1_n_explore_obj2'] = self.calc_action_n(1, ['obj_2'])
        self.featuredict['min2_n_explore_obj2'] = self.calc_action_n(2, ['obj_2'])
        self.featuredict['min3_n_explore_obj2'] = self.calc_action_n(3, ['obj_2'])
        self.featuredict['min4_n_explore_obj2'] = self.calc_action_n(4, ['obj_2'])
        self.featuredict['min5_n_explore_obj2'] = self.calc_action_n(5, ['obj_2'])

        # Exploretimes
        self.featuredict['min1_explore_time'] = self.calc_actiontime(1, ['obj_1', 'obj_2'])
        self.featuredict['min2_explore_time'] = self.calc_actiontime(2, ['obj_1', 'obj_2'])
        self.featuredict['min3_explore_time'] = self.calc_actiontime(3, ['obj_1', 'obj_2'])
        self.featuredict['min4_explore_time'] = self.calc_actiontime(4, ['obj_1', 'obj_2'])
        self.featuredict['min5_explore_time'] = self.calc_actiontime(5, ['obj_1', 'obj_2'])

        self.featuredict['min1_obj1_time'] = self.calc_actiontime(1, ['obj_1'])
        self.featuredict['min2_obj1_time'] = self.calc_actiontime(2, ['obj_1'])
        self.featuredict['min3_obj1_time'] = self.calc_actiontime(3, ['obj_1'])
        self.featuredict['min4_obj1_time'] = self.calc_actiontime(4, ['obj_1'])
        self.featuredict['min5_obj1_time'] = self.calc_actiontime(5, ['obj_1'])

        self.featuredict['min1_obj2_time'] = self.calc_actiontime(1, ['obj_2'])
        self.featuredict['min2_obj2_time'] = self.calc_actiontime(2, ['obj_2'])
        self.featuredict['min3_obj2_time'] = self.calc_actiontime(3, ['obj_2'])
        self.featuredict['min4_obj2_time'] = self.calc_actiontime(4, ['obj_2'])
        self.featuredict['min5_obj2_time'] = self.calc_actiontime(5, ['obj_2'])

        # Discrimination Index
        self.featuredict['min1_DI'] = self.calc_DI(1)
        self.featuredict['min2_DI'] = self.calc_DI(2)
        self.featuredict['min3_DI'] = self.calc_DI(3)
        self.featuredict['min4_DI'] = self.calc_DI(4)
        self.featuredict['min5_DI'] = self.calc_DI(5)

        # Bouts
        self.featuredict['bout_time'] = self.featuredict['min5_explore_time'] / (self.featuredict['min5_n_explore'] + np.exp(-100))
        self.featuredict['bout_obj1_time'] = self.featuredict['min5_obj1_time'] / (self.featuredict['min5_n_explore_obj1'] + np.exp(-100))
        self.featuredict['bout_obj2_time'] = self.featuredict['min5_obj2_time'] / (self.featuredict['min5_n_explore_obj2'] + np.exp(-100))










# CALCULATE FEATURES
# data = DataSet()
# for i in range(len(data)):
#     actions = data(i)
#     if actions:
#         animal = AnimalFeatures(actions)
#         animal.make_features()
#         data.save_features_in_df(i, animal.featuredict)



# SAVE ALL DFS AS ONE IN DATA
dfs =[]
files = os.listdir(DataSet.path_features)
for file in files:
    df = pd.read_csv(DataSet.path_features +'/' + file)
    dfs.append(df)

superdf = pd.concat(dfs)
superdf.to_csv(os.getcwd() + '/data/ehmt1/SS_alldata_autoscored.csv', index=False)

