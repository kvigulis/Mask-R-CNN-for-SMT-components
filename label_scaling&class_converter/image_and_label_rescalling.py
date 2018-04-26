import os
import json
from PIL import Image
from resizeimage import resizeimage

'''Used to downscale the images and labels'''

DATA_DIR = os.getcwd() + os.sep + 'mask_all_data'

# input directories
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# output directories
SCALED_IMAGES_DIR = os.path.join(DATA_DIR, "scaled_images")
SCALED_LABELS_DIR = os.path.join(DATA_DIR, "scaled_labels")

images_files = os.listdir(IMAGES_DIR)

class_name = 'SMT'
scaling_factor = 0.5

def label_scale_change_class_and_save(image_name, rescale):
    label_name = image_name[:-4] + '_labels.json'
    label_path = os.path.join(LABELS_DIR, label_name)
    scaled_labels_path = os.path.join(SCALED_LABELS_DIR, label_name)
    label_dict = json.load(open(label_path))
    label_obj_list = label_dict['labels']
    label_obj_list_new = []
    print(label_path)
    for label_obj in label_obj_list:
        # rescale
        if rescale:
            if label_obj['label_type'] == 'box':
                label_obj['centre']['x'] = label_obj['centre']['x'] * scaling_factor
                label_obj['centre']['y'] = label_obj['centre']['y'] * scaling_factor
                label_obj['size']['x'] = label_obj['size']['x'] * scaling_factor
                label_obj['size']['y'] = label_obj['size']['y'] * scaling_factor
            if label_obj['label_type'] == 'polygon':
                for vertex in label_obj['vertices']:
                    vertex['x'] = vertex['x'] * scaling_factor
                    vertex['y'] = vertex['y'] * scaling_factor
        # set class name
        label_obj['label_class'] = class_name
        label_obj_list_new.append(label_obj)
    label_dict['labels'] = label_obj_list_new
    with open(scaled_labels_path, 'w') as file:
        json.dump(label_dict, file)



images_changed = 0
for idx, image in enumerate(images_files):
    image_path = os.path.join(IMAGES_DIR, image)
    scaled_image_path = os.path.join(SCALED_IMAGES_DIR, image)
    with open(image_path, 'r+b') as f:
        with Image.open(f) as pil_image:
            width, height = pil_image.size
            print("width: ",width, "; height: ", height)
            #resize only if largest dimension is larger than max_dim setting in Mask R-CNN
            if max([width,height]) > 1408:
                images_changed += 1
                print("Number of images and labels rescaled: ", images_changed)
                try:
                    label_scale_change_class_and_save(image, True)
                except:
                    print("label not found")
                pil_image = resizeimage.resize_cover(pil_image, [width*scaling_factor, height*scaling_factor])
                pil_image.save(scaled_image_path, pil_image.format)
            else:
                try:
                    label_scale_change_class_and_save(image, False)
                except:
                    print("label not found")
                pil_image.save(scaled_image_path, pil_image.format)



