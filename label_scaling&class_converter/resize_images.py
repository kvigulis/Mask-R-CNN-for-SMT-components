import os
import json
from PIL import Image
from resizeimage import resizeimage

'''Used to downscale the images'''

root = os.getcwd()

images_dir = os.path.join(root, "images")
scaled_labels_dir = os.path.join(root, "scaled_images")

images_files = os.listdir(images_dir)
scaling_factor = 0.5

images_changed = 0
for idx, image in enumerate(images_files):
    
    image_path = os.path.join(images_dir, image)
    scaled_image_path = os.path.join(scaled_labels_dir, image)
    with open(image_path, 'r+b') as f:
        with Image.open(f) as pil_image:
            width, height = pil_image.size
            print("width: ",width, "; height: ", height)
            #resize only if largest dimension is larger than max_dim setting in Mask R-CNN
            if max([width,height]) > 1408:
                images_changed += 1
                print(images_changed)
                pil_image = resizeimage.resize_cover(pil_image, [width*scaling_factor, height*scaling_factor])

            pil_image.save(scaled_image_path, pil_image.format)
