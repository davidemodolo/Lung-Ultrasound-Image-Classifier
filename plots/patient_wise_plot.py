import pandas as pd
# "patient_id exam_id spot[1-14] frame_number score"
# load patients data in a dataframe from images folder
import glob
images_paths = glob.glob("images/*.png", recursive=True)
# images are named as: patientid_examid_spotnumber_framenumber_score.png
# create a dataframe with the data removing "images/"
images_df = pd.DataFrame([path[7:-4].split("_") for path in images_paths], columns=["patient_id", "exam_id", "spot", "frame_number", "score"])
images_df["score"] = images_df["score"].astype(str)
images_df["frame_number"] = images_df["frame_number"].astype(str)
images_df["spot"] = images_df["spot"].astype(str)
images_df["patient_id"] = images_df["patient_id"].astype(str)
images_df["exam_id"] = images_df["exam_id"].astype(str)

import matplotlib.pyplot as plt
import seaborn as sns
# sort rows by score
images_df = images_df.sort_values(by=["score"])
# plot the number of 0,1,2,3 scores for each patient with the value on top of the bar
plt.figure(figsize=(20,10))
ax = sns.countplot(x="patient_id", hue="score", data=images_df)
ax.set_title("Number of frames by score for each patient")
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x(), p.get_height() * 1.005))
plt.show()


# plot the overall number of 0,1,2,3 scores with the value on top of the bar
plt.figure(figsize=(20,10))
ax = sns.countplot(x="score", data=images_df)
ax.set_title("Overall number frames by score")
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.show()

# print the number of frames for each patient
print(images_df.groupby("patient_id").count()["score"])