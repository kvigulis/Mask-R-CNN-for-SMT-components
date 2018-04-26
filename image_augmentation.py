import os
import json
import cv2
import numpy as np
import scipy.misc
from PIL import Image
from PIL import ImageFilter

'''
Image augmentation script:
Labels have to be in UEA labelling tool format.

Author: K. Vigulis
'''


ROOT_DIR = os.getcwd()

# Input directories
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
LABEL_DIR = os.path.join(ROOT_DIR, "labels")

IMAGE_FILES = os.listdir(IMAGE_DIR)

# Output directories
OUTPUT_IMAGE_DIR = os.path.join(ROOT_DIR, "output_images")
OUTPUT_LABEL_DIR = os.path.join(ROOT_DIR, "output_labels")

def horizontal_flip(image_np,labels):
    # Flip image and labels horizonatlly
    modified_image = np.fliplr(image_np)
    #modified_image = np.flip(modified_image, 2)
    modified_labels = []
    for label in labels:

        if label['label_type'] == 'box':
            # Readjust the box centers. The size stays the same in this case.
            print(label['centre'])
            label['centre'] = {'x': (width - label['centre']['x'] ), 'y': label['centre']['y']}
            print(label['centre'], "\n\n")
        else:
            #print(label)
            modified_vertices = []
            for vertex in label['vertices']:
                vertex = {'x': (width - vertex['x']), 'y': vertex['y']}
                modified_vertices.append(vertex)
            label['vertices'] = modified_vertices

        modified_labels.append(label)
    #print("within function",modified_labels[0])
    return modified_image, modified_labels

def vertical_flip(image_np,labels):
    # Flip image and labels vertically
    modified_image = np.flipud(image_np)
    #modified_image = np.flip(modified_image, 2)
    modified_labels = []
    for label in labels:

        if label['label_type'] == 'box':
            # Readjust the box centers. The size stays the same in this case.
            print(label['centre'])
            label['centre'] = {'x': label['centre']['x'], 'y': (height - label['centre']['y'])}
            print(label['centre'], "\n\n")
        else:
            #print(label)
            modified_vertices = []
            for vertex in label['vertices']:
                vertex = {'x': vertex['x'], 'y': (height - vertex['y'])}
                modified_vertices.append(vertex)
            label['vertices'] = modified_vertices

        modified_labels.append(label)
    #print("within function",modified_labels[0])
    return modified_image, modified_labels


def awgnoise(image_np,labels):
    # Additive white gaussian noise
    noise = np.random.normal(-40, +40, image_np.shape)
    modified_image = image_np + noise
    # Make sure that color values stay in range
    modified_image = np.clip(modified_image, 0, 255)
    return modified_image, labels

for idx, image in enumerate(IMAGE_FILES):
    #image_np = cv2.imread(os.path.join(IMAGE_DIR,image))
    image_PIL = Image.open(os.path.join(IMAGE_DIR,image))

    # Add color overlay
    #layer = Image.new('RGB', image_PIL.size, 'blue')
    #image_PIL = Image.blend(image_PIL, layer, 0.3)

    # Choose the filter to apply to the image.
    #image_PIL = image_PIL.filter(ImageFilter.BLUR)
    image_PIL = image_PIL.filter(ImageFilter.EDGE_ENHANCE)
    image_PIL = image_PIL.filter(ImageFilter.SMOOTH)
    image_PIL = image_PIL.filter(ImageFilter.SHARPEN)



    image_np =np.asarray(image_PIL)
    height, width, channels = image_np.shape
    print(image_np.shape)
    label_name = image.split(".")[0] + "_labels.json"
    try:
        labels_json = json.load(open(os.path.join(LABEL_DIR, label_name)))
        output_labels_json = labels_json
        labels = labels_json['labels']
        print("before mod:", labels)
    except:
        print("Warning: label for image ", image, " not found.")
        labels = []


    # Uncomment and comment for the desired flips.
    # Horizontal
    #modified_image, modified_labels = horizontal_flip(image_np, labels)
    # Vertical
    #modified_image, modified_labels = vertical_flip(image_np, labels)

    # Diagonal flip
    #modified_image, modified_labels = horizontal_flip(image_np, labels)
    #modified_image, modified_labels = vertical_flip(modified_image, modified_labels)

    # Additive White Gaussian Noise
    modified_image, modified_labels = awgnoise(image_np, labels)
    #print("labels:", labels, "\nmodified labels:", modified_labels, "\n\n\n")


    # Uncomment to save(move to output directory) the orginal, unchanged images and labels. Should do this at least once
    #modified_image, modified_labels = image_np, labels


    # set the image and label modification name depending on the modifications
    modification_suffix = "_sh_awgn"

    modified_image_name = image.split(".")[0] + modification_suffix + ".jpg"
    ouput_image_path = os.path.join(OUTPUT_IMAGE_DIR, modified_image_name)
    scipy.misc.imsave(ouput_image_path, modified_image)


    output_labels_json['image_filename'] = modified_image_name
    output_labels_json['labels'] = labels
    modifiel_label_name = image.split(".")[0] + modification_suffix + "_labels.json"
    ouput_label_path = os.path.join(OUTPUT_LABEL_DIR, modifiel_label_name)
    with open(ouput_label_path, 'w') as outfile:
        json.dump(output_labels_json, outfile)




