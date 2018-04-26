import os
import json

'''Converts all the label classes to component'''

root = os.getcwd()

labels_dir = os.path.join(root, "labels")
converted_labels_dir = os.path.join(root, "converted_labels")

labels_files = os.listdir(labels_dir)

for label in labels_files:
    label_path = os.path.join(labels_dir, label)
    converted_labels_path = os.path.join(converted_labels_dir, label)
    label_dict = json.load(open(label_path))
    label_obj_list = label_dict['labels']
    label_obj_list_new = []
    for label_obj in label_obj_list:
        label_obj['label_class'] = 'person on motorcycle'
        label_obj_list_new.append(label_obj)
    label_dict['labels'] = label_obj_list_new
    print(label_dict)
    with open(converted_labels_path, 'w') as file:
        json.dump(label_dict, file)
print(len(labels_files))
