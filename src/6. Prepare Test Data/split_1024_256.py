"""
Split data into 256x256
(c)2021 Arif Ahmed Sekh
"""

import os
import glob
import cv2
from PIL import Image, ImageFilter 
import numpy as np
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None # to avoid image size warning

imgdir='png'
savedir='data'

Path(savedir).mkdir(parents=True, exist_ok=True)
 
start_pos = start_x, start_y = (0, 0)
cropped_image_size = w, h = (256, 256)
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
for file in filelist:
    img = Image.open(file)
    width, height = img.size

    frame_num = 1
    for col_i in range(0, width, w):
        for row_i in range(0, height, h):
            crop = img.crop((col_i, row_i, col_i + w, row_i + h))
            name = os.path.basename(file)
            name = os.path.splitext(name)[0]
            save_to= os.path.join(savedir, name+"_{:03}.png")
            crop.save(save_to.format(frame_num))
            frame_num += 1