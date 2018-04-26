import os
import json

'''Used to downscale the polygon coordinates'''

root = os.getcwd()

labels_dir = os.path.join(root, "converted_labels")
scaled_labels_dir = os.path.join(root, "scaled_labels")

labels_files = os.listdir(labels_dir)
#Scaling factor for labels. Should correspond to images scaling factor.
scaling_factor = 0.5

for label in labels_files:
    label_path = os.path.join(labels_dir, label)
    scaled_labels_path = os.path.join(scaled_labels_dir, label)
    label_dict = json.load(open(label_path))
    label_obj_list = label_dict['labels']
    label_obj_list_new = []
    print(label_path)
    for label_obj in label_obj_list:
        if label_obj['label_type'] == 'box':
            label_obj['centre']['x'] = label_obj['centre']['x']*scaling_factor
            label_obj['centre']['y'] = label_obj['centre']['y']*scaling_factor
            label_obj['size']['x'] = label_obj['size']['x']*scaling_factor
            label_obj['size']['y'] = label_obj['size']['y']*scaling_factor
        if label_obj['label_type'] == 'polygon':
            for vertex in label_obj['vertices']:
                vertex['x'] = vertex['x']*scaling_factor
                vertex['y'] = vertex['y']*scaling_factor

        label_obj_list_new.append(label_obj)
        label_dict['labels'] = label_obj_list_new
        with open(scaled_labels_path , 'w') as file:
            json.dump(label_dict, file)
    print(label_obj_list)
