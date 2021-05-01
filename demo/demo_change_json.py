"""
Separate day and night into two annotation files
"""

import pdb
import os
import json
from os.path import isfile, join
import cv2

data_set = 'train'
in_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_annotations.json'#thermal_annotations_4_channel_no_dogs.json'
out_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_RGBT_pairs_3_class.json'
img_folder = '../../../Datasets/FLIR/'+data_set+'/RGB'

#in_anno_file = out_anno_file
data = json.load(open(in_anno_file, 'r'))
info = data['info']
categories = data['categories']
licenses = data['licenses']
annos = data['annotations']
images = data['images']
pdb.set_trace()
annotations = []
# Remove dog categories
for i in range(len(annos)):
    if annos[i]['category_id'] == 17: # dog
        continue
    else:
        annos[i]['category_id'] -= 1
        annotations.append(annos[i])

file_names = [f for f in os.listdir(img_folder) if isfile(join(img_folder, f))]

rgb_img_dict = {}

for i in range(len(file_names)):
    img_id = int(file_names[i].split('FLIR_')[1].split('.')[0])
    rgb_img_dict[img_id] = file_names[i]

# Create new image list and annotation lists
annos_new = []
images_new = []
anno_cnt = 0
for i in range(len(images)):
    img_name = images[i]['file_name']
    #img_name = contents[i]
    img_file_num = int(img_name.split('FLIR_')[1].split('.')[0])
    
    if img_file_num in rgb_img_dict.keys():
        img_info = images[i].copy()
        images_new.append(img_info)        
        img_id = images[i]['id']
        # Skip annotations with images that are not in this image ID
        while annotations[anno_cnt]['image_id'] < img_id:
            anno_cnt += 1
        
        # Record annotations in the required image ID
        while annotations[anno_cnt]['image_id'] == img_id:
            annos_new.append(annotations[anno_cnt])
            anno_cnt += 1
            if anno_cnt == len(annotations):
                break
            print('image id = ', img_id, 'anno id = ', anno_cnt)

for i in range(3):
    categories[i]['id'] -= 1

out_json = {}
out_json['info'] = info
out_json['categories'] = categories[:3]
out_json['licenses'] = licenses
out_json['annotations'] = annos_new
out_json['images'] = images_new

with open(out_anno_file, 'w') as outfile:
    json.dump(out_json, outfile, indent=2)
