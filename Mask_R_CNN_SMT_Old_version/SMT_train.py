import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import json
import sys
import random
import math
import re
import time
import numpy as np
import cv2

from config import Config
import utils
import model as modellib
import visualize
from model import log
from pycocotools import mask as maskUtils

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from coco import CocoDataset


coco_obj = CocoDataset()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "smt"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1408

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128
        
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8
    
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 300
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 150    
    
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.0008
    LEARNING_MOMENTUM = 0.9
    
    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when inferencing
    TRAIN_BN = None  # Defaulting to False since batch size is often small
    
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 30

config = ShapesConfig()


class SMTDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, IMG_PATH, LBL_PATH):
        """
        Loads images from /images/
        """
        # Add classes
        self.add_class("Aerial", 1, "SMT")        
        self.add_class("Aerial", 2, "Unclassified" )

        # Add images
        onlyfiles = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]
        #onlylabels = [f for f in listdir(LBL_PATH) if isfile(join(LBL_PATH, f))]
        
        
        for i in onlyfiles:
            IMAGE_PATH = os.path.join(IMG_PATH, i)
            IMAGE_LABEL_PATH = LBL_PATH + "/" + i[:-4] + "_labels.json"            
            
            
            #IMAGE_LABEL_PATH = os.path.join(LBL_PATH, i)
            #image_name = i[:-12] + ".jpg"
            #IMAGE_PATH = os.path.join(IMG_PATH, image_name)
            print(IMAGE_PATH)
            try:
                json_annotaion_file=open(IMAGE_LABEL_PATH).read()                
                annotation = json.loads(json_annotaion_file)   
            except:
                # if the *.json label file is not found add empty annotation. E.g. in case of test images.
                print("WARNING: Image annotation not found")
                annotation = {"image_filename": i, "complete": None, "labels": []}
            
            print(IMAGE_PATH)
            with Image.open(IMAGE_PATH) as img:
                width, height = img.size
            
            self.add_image("Aerial", image_id=i, path=IMAGE_PATH, width=width, height=height, annotation=annotation)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "smt":
            return info["smt"]
        else:
            super(self.__class__).image_reference(self, image_id)

            
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks."""
        
        
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]        

        instance_masks = []
        class_ids = []             
        annotations = self.image_info[image_id]['annotation']['labels'] 
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.    
        class_map = {"SMT":1, "Unclassified":2} 
        
        for annotation in annotations:            
            #class_id = annotation["label_class"] 
            
            if annotation["label_class"] == None:
                class_id = class_map["Unclassified"]                
            else:
                class_id = class_map[annotation["label_class"]]
            
            
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                
                instance_masks.append(m)
                class_ids.append(class_id)        
        # Pack instance masks into an array    
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
#             print("No labels")
#             # Call super class to return an empty mask
#             mask = np.array(instance_masks, dtype=np.int32)            
#             class_ids = np.array(class_ids, dtype=np.int32)
#             print("Class Ids", class_ids)
#             return mask, class_ids
            return super(CocoDataset, coco_obj).load_mask(image_id)
            
        
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if ann['label_type'] == "box":
            centre_x_y = ann['centre']            
            size_x_y = ann['size']                
            
            segm = [centre_x_y['x'] - size_x_y['x']/2,centre_x_y['y'] + size_x_y['y']/2,
                   centre_x_y['x'] - size_x_y['x']/2,centre_x_y['y'] - size_x_y['y']/2,
                   centre_x_y['x'] + size_x_y['x']/2,centre_x_y['y'] - size_x_y['y']/2,
                   centre_x_y['x'] + size_x_y['x']/2,centre_x_y['y'] + size_x_y['y']/2,]
            segm = [segm] # make list within list.
        else:   
            segm_x_y = ann['vertices'] # Comes in a format [{'x': 464.5543363223196, 'y': 458.5734663855463},...}
            # Need to convert 'segm_x_y' to an even number of floats in a list for the 'maskUtils' code to work.
            segm = []
            for vertex in segm_x_y:            
                segm.append(vertex['x'])
                segm.append(vertex['y'])
            segm = [segm] # 'segm' in [[227.41, 81.56, 312.81, 91.16, ....]] format.        
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            try:
                rles = maskUtils.frPyObjects(segm, height, width)
                rle = maskUtils.merge(rles)
            except:
                print("Error with object: ", ann)                
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


#Load the datasets
TRAIN_DIR = os.path.join(ROOT_DIR, "train7360/images")
TRAIN_LBL_DIR = os.path.join(ROOT_DIR, "train7360/labels")

VAL_DIR = os.path.join(ROOT_DIR, "val7360/images")
VAL_LBL_DIR = os.path.join(ROOT_DIR, "val7360/labels")




# Training dataset
dataset_train = SMTDataset()
dataset_train.load_shapes(TRAIN_DIR, TRAIN_LBL_DIR)
dataset_train.prepare()


# Validation dataset
dataset_val = SMTDataset()
dataset_val.load_shapes(VAL_DIR, VAL_LBL_DIR)
dataset_val.prepare()






# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
    #model.load_weights(os.path.join(ROOT_DIR, "mask_rcnn_smt_0021.h5"), by_name=True)



# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_train, 
            learning_rate=config.LEARNING_RATE, 
            epochs=67,    
            layers='heads')
