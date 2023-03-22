import os
import glob
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFilter
from PIL import ImageOps
from tqdm import tqdm


# get all png files in images
files = glob.glob('images/*.png')

sizes = {}
for file in tqdm(files):
    patient = os.path.basename(file).split('_')[0]
    img = Image.open(file)
    size = img.size
    if size in sizes:
        sizes[size][0] += 1
        sizes[size][1].append(patient)
    else:
        sizes[size] = [1, [patient]]

# print sizes
print("\n ImgSize\t Count\t Patients")
for size in sizes:
    print("\n", size, "\t", sizes[size][0], "\t", set(sizes[size][1]))

print("\n")