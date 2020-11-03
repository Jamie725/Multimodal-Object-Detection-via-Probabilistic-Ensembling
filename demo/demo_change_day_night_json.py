"""
Separate day and night into two annotation files
"""

import pdb
import os
import json
from os.path import isfile, join
import cv2

time = 'Night'
data_set = 'val'
in_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_annotations_4_channel_no_dogs.json'
out_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_annotations_4_channel_no_dogs_'+time+'.json'
img_folder = '../../../Datasets/FLIR/'+data_set+'/RGB'



#in_anno_file = out_anno_file
data = json.load(open(in_anno_file, 'r'))
info = data['info']
categories = data['categories']
licenses = data['licenses']
annos = data['annotations']
images = data['images']

annotations = []
file_names = [f for f in os.listdir(img_folder) if isfile(join(img_folder, f))]

rgb_img_dict = {}
image_name_list = '../../../Datasets/FLIR/'+data_set+'/RGB/'+time+'/'+time+'_name_list.txt'
with open(image_name_list) as f:
    contents = f.readlines()

for i in range(len(contents)):
    img_name = contents[i]
    img_id = int(img_name.split('FLIR_')[1].split('.')[0])
    rgb_img_dict[img_id] = img_name

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
        """
        fname = img_info['file_name'].split('.jpeg')[0]
        fname = 'RGB/' + fname.split('8_bit/')[1] + '.jpg'
        img_info['file_name'] = fname
        img = cv2.imread('../../../Datasets/FLIR/'+data_set+'/'+fname)

        img_info['height'] = img.shape[0]
        img_info['width'] = img.shape[1]
        """
        images_new.append(img_info)        
        img_id = images[i]['id']
        # Skip annotations with images that are not in this image ID
        while annos[anno_cnt]['image_id'] < img_id:
            anno_cnt += 1
        
        # Record annotations in the required image ID
        while annos[anno_cnt]['image_id'] == img_id:
            annos_new.append(annos[anno_cnt])
            anno_cnt += 1
            if anno_cnt == len(annos):
                break
            print('image id = ', img_id, 'anno id = ', anno_cnt)
        
out_json = {}
out_json['info'] = info
out_json['categories'] = categories
out_json['licenses'] = licenses
out_json['annotations'] = annos_new
out_json['images'] = images_new

with open(out_anno_file, 'w') as outfile:
    json.dump(out_json, outfile, indent=2)
