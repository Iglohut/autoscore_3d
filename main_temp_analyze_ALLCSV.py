from PredictFeatures.PredictActions import *


fails = []
for i in range(len(SequenceExtracter.df_all)):

    myvid = SequenceExtracter(i)

    if myvid.canalyze:
        print("Analyzing video({}): {}".format(i, myvid.vidname))
        myvid.get_pivot_locations()
        myvid.get_actions()
        myvid.save_actions()
        myvid.save_status(status=3)
    else:
        print("Couldn't analyze video({})".format(i))
        fails.append(i)