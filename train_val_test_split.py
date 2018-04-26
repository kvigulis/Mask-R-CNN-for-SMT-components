import os
import json
import random
from PIL import Image

'''Split the dataset into three parts'''


DATA_DIR = os.getcwd() + os.sep + '7360_images_with_augmentation'

# input directories
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# train output directories
TRAIN_DIR = DATA_DIR + os.sep + 'train7360'
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "images")
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, "labels")

# validation output directories
VAL_DIR = DATA_DIR + os.sep + 'val7360'
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
VAL_LABELS_DIR = os.path.join(VAL_DIR, "labels")

# test output directories
TEST_DIR = DATA_DIR + os.sep + 'test7360'
TEST_IMAGES_DIR = os.path.join(TEST_DIR, "images")
TEST_LABELS_DIR = os.path.join(TEST_DIR, "labels")


images_files_list = os.listdir(IMAGES_DIR)

dataset_size = len(images_files_list)


# What fraction of the whole data to allocated for evaluation sets (val and test)
fraction_for_evaluation_data = 0.1

# What fraction of the evaluation data to allocated for the validation set
fraction_for_val_set = 0.75

# number of samples in val + test sets
evaluation_size = int(dataset_size*fraction_for_evaluation_data)
# number of samples in val set
val_size = int(evaluation_size*fraction_for_val_set)


# Split datasets
evaluation_images_list = random.sample(images_files_list, evaluation_size)
val_images_list = random.sample(evaluation_images_list, val_size)
test_images_list = [image_name for image_name in evaluation_images_list if image_name not in val_images_list]
train_images_list = [image_name for image_name in images_files_list if image_name not in evaluation_images_list]

# Equation should be satisfied
print("[DEBUG]", len(val_images_list)+len(test_images_list), " = ", len(evaluation_images_list))
print("[DEBUG]", len(train_images_list)+len(evaluation_images_list), " = ", len(images_files_list))


def save_set(image_name_list, output_dir):
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    for idx, image_name in enumerate(image_name_list):

        # Save image
        image_path = os.path.join(IMAGES_DIR, image_name)
        image_output_path = os.path.join(output_images_dir, image_name)
        with open(image_path, 'r+b') as f:
            with Image.open(f) as pil_image:
                pil_image.save(image_output_path, pil_image.format)

        # Save label
        label_name = image_name[:-4] + '_labels.json'
        label_path = os.path.join(LABELS_DIR, label_name)
        label_dict = json.load(open(label_path))
        output_labels_path = os.path.join(output_labels_dir, label_name)
        with open(output_labels_path, 'w') as file:
            json.dump(label_dict, file)


# Save datasets in corresponding directories
save_set(train_images_list, TRAIN_DIR)
print("Train set DONE!")
save_set(val_images_list, VAL_DIR)
print("Val set DONE!")
save_set(test_images_list, TEST_DIR)
print("Test set DONE!")