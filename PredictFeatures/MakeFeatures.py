import pandas as pd
import csv
import os

class DataSet:
    path_overview = os.getcwd() + '/data/ehmt1/VideoNamesStatus.csv'
    path_actions = os.getcwd() + '/data/ehmt1/ehmt1_actions'
    path_features = os.getcwd() + '/data/ehmt1/ehmt1_features'

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


    def save_features_in_df(self, i, features):
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
        for feature in list(featureset.keys()):
            df[feature] = featureset[feature]

        # Save it
        path_df = self.path_features + '/' + df.VideoName.unique()[0].split('.avi')[0].split('.')[-1] + '.csv'
        df.to_csv(path_df, index=False)
        print("Feature file saved to:", path_df)



    def __call__(self, i):
        """
        :param i: row number
        :return: trial info df and action list
        """
        if i > len(self): # If row cannot exist
            return False
        # Load actions corresponding video
        actions = self.load_actions(self.df_overview.VideoName[i])
        return actions


    def __len__(self):
        """Length is length of all the rows in overview."""
        return len(self.df_overview)


data = DataSet()

actions = data(1)

# TODO Check save features
# TODO make AnimalFeature class

features = ["first_object", "first_object_latency", "stay1", "stay2", "SS1", "perseverance", "n_transitions",
            "min1_n_explore", "min2_n_explore", "min3_n_explore", "min4_n_explore", "min5_n_explore",
            'min1_n_explore_obj1', 'min2_n_explore_obj1', 'min3_n_explore_obj1', 'min4_n_explore_obj1', 'min5_n_explore_obj1',
            'min1_n_explore_obj2', 'min2_n_explore_obj2', 'min3_n_explore_obj2', 'min4_n_explore_obj2', 'min5_n_explore_obj2',
            "min1_obj1_time", "min2_obj1_time", "min3_obj1_time", "min4_obj1_time", "min5_obj1_time",
            "min1_obj2_time", "min2_obj2_time", "min3_obj2_time", "min4_obj2_time", "min5_obj2_time",
            "min1_DI", "min2_DI", "min3_DI", "min4_DI", "min5_DI",
            "min1_explore_time", "min2_explore_time", "min3_explore_time", "min4_explore_time", "min5_explore_time",
            "bout_time", "bout_obj1_time", "bout_obj2_time"]

featureset = dict.fromkeys(features)