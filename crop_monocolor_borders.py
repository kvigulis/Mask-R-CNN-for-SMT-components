import os
import json
from PIL import Image, ImageChops
from resizeimage import resizeimage

root = os.getcwd()

images_dir = os.path.join(root, "predictions")
scaled_labels_dir = os.path.join(root, "cropped_predictions")

images_files = os.listdir(images_dir)


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)



images_changed = 0
for idx, image in enumerate(images_files):
    image_path = os.path.join(images_dir, image)
    scaled_image_path = os.path.join(scaled_labels_dir, image)
    with open(image_path, 'r+b') as f:
        with Image.open(f) as pil_image:
            im = trim(pil_image)
            im = trim(im)

            im.save(scaled_image_path, pil_image.format, quality=100)

