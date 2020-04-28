import pdb
import os
import json
from os.path import isfile, join

data_set = 'val'
in_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_annotations_new.json'
out_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_annotations_4_channel_no_dogs.json'
img_folder = '../../../Datasets/FLIR/'+data_set+'/RGB'
#in_anno_file = out_anno_file
data = json.load(open(in_anno_file, 'r'))
info = data['info']
categories = data['categories']
licenses = data['licenses']
annos = data['annotations']
images = data['images']

annotations = []
# Remove dog categories
for i in range(len(annos)):
    if annos[i]['category_id'] == 16: # dog
        continue
    else:
        annotations.append(annos[i])
    
file_names = [f for f in os.listdir(img_folder) if isfile(join(img_folder, f))]

rgb_img_dict = {}

for i in range(len(file_names)):
    img_name = file_names[i]
    img_id = int(img_name.split('_')[1].split('.')[0])

    rgb_img_dict[img_id] = img_name

# Create new image list and annotation lists
annos_new = []
images_new = []
anno_cnt = 0
for i in range(len(images)):
    img_name = images[i]['file_name']
    img_file_num = int(img_name.split('FLIR_')[1].split('.')[0])
    if img_file_num in rgb_img_dict.keys():
        images_new.append(images[i])

        img_id = images[i]['id']
        
        while annotations[anno_cnt]['image_id'] < img_id:
            anno_cnt += 1
        
        while annotations[anno_cnt]['image_id'] == img_id:
            annos_new.append(annotations[anno_cnt])
            anno_cnt += 1
            if anno_cnt == len(annotations):
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
