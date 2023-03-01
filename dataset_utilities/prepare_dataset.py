import scipy.io
import glob
import cv2
from tqdm import tqdm
dotmat_paths = glob.glob("**/*.mat", recursive=True)
frames_paths = [x for x in dotmat_paths if "_score" not in x]
scores_paths = [x for x in dotmat_paths if "_score" in x]
frames_paths.sort(key=lambda x: x.split("_")[-1])
scores_paths.sort(key=lambda x: x.split("_")[-2])

for frames, scores in zip(frames_paths, scores_paths):
    file_name = frames.split("/")[-1].split(".")[0].split("_")
    scores_name = scores.split("/")[-1].split(".")[0].split("_")
    patient_id = file_name[1]
    exam_id = file_name[2]
    spot_number = file_name[3]

    labels = scipy.io.loadmat(scores)["Score_matrix"].sum(axis=0)

    frames = scipy.io.loadmat(frames)["frames"]
    frame_num = frames.shape[3]
    
    for i in range(frame_num):
        if i == frame_num-1 and frame_num!=len(labels):
            continue
        frame = frames[:,:,:,i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(f"{patient_id}_{exam_id}_{spot_number}_{i}_{labels[i]}")
        cv2.imwrite(f"images_png/{patient_id}_{exam_id}_{spot_number}_{i}_{labels[i]}.png", frame)
