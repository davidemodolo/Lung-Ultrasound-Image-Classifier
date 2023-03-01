# get all png files in images to size [1, 512, 512]
# save to images_resized
import os
import glob
import numpy as np
from PIL import Image
# Use Resampling.LANCZOS instead of ANITALIAS
# to avoid blurring
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFilter
from PIL import ImageOps


# get all png files in images
files = glob.glob('images/*.png')

# resize images to [1, 512, 512]
for file in files:
    img = Image.open(file)
    img = img.resize((512, 512), Image.LANCZOS)
    img.save('images_resized/' + os.path.basename(file))

