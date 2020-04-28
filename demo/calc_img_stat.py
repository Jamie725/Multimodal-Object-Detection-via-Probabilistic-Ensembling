# import some common detectron2 utilities
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os
import pdb

# get path
#mypath = 'input/FLIR/Day/'
dataset = 'FLIR'
mode = 'train'
img_folder = '../../../Datasets/FLIR/train/thermal_8_bit/'

files_names = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]

img_means = []
img_stds = []
for i in range(len(files_names)):
    # Read image file
    path = img_folder + files_names[i]
    img = cv2.imread(path)

    # Get statistics
    mean = np.mean(img[:,:,0])
    std = np.std(img[:,:,0])
    img_means.append(mean)
    img_stds.append(std)

print('Thermal image mean = ', np.mean(img_means))
print('Thermal image std = ', np.mean(img_stds))