import pandas as pd
from pathlib import Path
import re
import os
from OS_utils import _vidlength

def name2seq(name):
    """

    :param name: raw video name
    :return: sequence number of that video
    """
    return int(name.split("_t0")[-1].split("_")[0])

def name2subjects(name):
    """

    :param name: raw video name
    :return: range of animal subject numbers corresponding to the batch of that video
    """
    numbers = name.split("_t0")[0].split("_")[-2:]
    numbers = [int(re.search(r'\d+', n).group()) for n in numbers]
    if numbers[0] > 100:
        numbers[-1] = int(str(numbers[0])[0] + str(numbers[-1]))
    return range(numbers[0], numbers[-1]+1)


# Create main data frame
VideoNamesPath = Path("./PredictUtils/VideoNames.txt").resolve()
VideoNamesSatusPath = str(VideoNamesPath).split('.txt')[0] + "Status.csv"
VideoNames = [line.rstrip('\n') for line in open(VideoNamesPath)]

columns = ["VideoName","StatusPredicted", "framelength", "sequence_nr", "subject", "genotype", "condition","trial", "obj_1", "obj_2"]

df = pd.DataFrame(columns=columns)
df["VideoName"] = VideoNames
df["StatusPredicted"] = df["StatusPredicted"].fillna(0).astype(int) # with 0s rather than NaNs

# Information file
df_help = pd.read_csv(Path("PredictUtils/SS_alldata.csv").resolve())

# Fill in the blanks
for i, row in df.iterrows():
    absname = row.VideoName
    sequence_i = name2seq(absname)
    subjects_i = name2subjects(absname)

    df_help_i = df_help.loc[(df_help['subject'].isin(subjects_i))]
    animal_i = df_help_i.loc[df_help_i['sequence_nr'] == sequence_i]

    if len(animal_i != 0):  # Only if the row exists in the summary data
        df.at[i, "sequence_nr"] = sequence_i
        df.at[i, "subject"] = animal_i.subject.values[0]
        df.at[i, "genotype"] = animal_i.genotype.values[0]
        df.at[i, "condition"] = animal_i.condition.values[0]
        df.at[i, "trial"]= animal_i.trial.values[0]
        df.at[i, "obj_1"] = animal_i.obj_1.values[0]
        df.at[i, "obj_2"] = animal_i.obj_2.values[0]
    else:
        df.at[i, "sequence_nr"] = sequence_i
    vidname_i = os.getcwd().split("autoscore_3d")[0] + "Intellectual_Disability/Intellectual_Disability" + absname[1:]
    df.at[i,"framelength"] = _vidlength(vidname_i)


df.to_csv(VideoNamesSatusPath, index=False)

dff = pd.read_csv("/media/iglohut/MD_Smits/Internship/autoscore_3d/PredictUtils/VideoNamesStatus.csv")