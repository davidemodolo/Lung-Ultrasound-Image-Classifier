# compute mean and standard deviation of a dataset contained in the images folder

import os
import cv2
import numpy as np
from tqdm import tqdm

# define the path to the images directory
IMAGES_PATH = "../images_png"

# initialize the list of means and standard deviations
means = np.zeros(3)
stds = np.zeros(3)
counter = 0

# loop over the image paths
for imagePath in tqdm(os.listdir(IMAGES_PATH)):
    # load the image, convert it to grayscale, and compute the
    # mean and standard deviation of each channel
    image = cv2.imread(os.path.join(IMAGES_PATH, imagePath))
    means += np.mean(image, axis=(0, 1))
    stds += np.std(image, axis=(0, 1))
    counter += 1

# compute the mean and standard deviation of the image
means /= counter
stds /= counter

# show the computed statistics in the format of transform.Normalize([x, x, x], [x, x, x])
print("mean = {}".format(means))
print("std = {}".format(stds))

