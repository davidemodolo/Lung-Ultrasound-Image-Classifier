# compute mean and standard deviation of a dataset contained in the images folder
# to run, put this file in the main folder

import os
import cv2
import numpy as np
from tqdm import tqdm

# define the path to the images directory
IMAGES_PATH = "images"

# initialize the list of means and standard deviations
means_r = []
means_g = []
means_b = []
stds_r = []
stds_g = []
stds_b = []
counter = 0

# loop over the image paths
for imagePath in tqdm(os.listdir(IMAGES_PATH)):
    # compute the mean and standard deviation of each channel
    image = cv2.imread(os.path.join(IMAGES_PATH, imagePath))
    means_b.append(np.mean(image[:, :, 0]))
    means_g.append(np.mean(image[:, :, 1]))
    means_r.append(np.mean(image[:, :, 2]))
    stds_b.append(np.std(image[:, :, 0]))
    stds_g.append(np.std(image[:, :, 1]))
    stds_r.append(np.std(image[:, :, 2]))
    counter += 1

# compute the mean and standard deviation of the images
means = np.zeros(3)	
stds = np.zeros(3)

means[0] = np.mean(means_r)
means[1] = np.mean(means_g)
means[2] = np.mean(means_b)
stds[0] = np.mean(stds_r)
stds[1] = np.mean(stds_g)
stds[2] = np.mean(stds_b)

# show the computed statistics in the format of transform.Normalize([x, x, x], [x, x, x])
print("mean = {}".format(means))
print("std = {}".format(stds))
print("transforms.Normalize(mean = {}, std = {})".format(means, stds))

