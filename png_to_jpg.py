import os
from PIL import Image
import numpy as np
import math


root = os.getcwd()
image_dir = os.path.join(root, "png_to_jpg")
imagefiles = os.listdir(image_dir)

for idx, i in enumerate(imagefiles):
    img = Image.open("png_to_jpg/" + i)
    img = img.convert("RGB")
    img.save('jpg/'+str(idx+1)+'pcb.jpg')
