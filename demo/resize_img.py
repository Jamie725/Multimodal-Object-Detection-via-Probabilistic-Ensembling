# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb

# get path
#mypath = 'input/FLIR/Day/'
dataset_name = 'FLIR'
data_set = 'val'
path = '../../../../Datasets/'+ dataset_name +'/'+data_set+'/RGB/'

files_names = [f for f in listdir(path) if isfile(join(path, f))]
#files_names = path + 'FLIR_05527.jpg'
out_folder = '../../../../Datasets/'+ dataset_name +'/'+data_set+'/resized_RGB/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

for i in range(len(files_names)):
    # get image
    file_img = path + files_names[i]
    #print('file = ',file_img)
    
    img = cv2.imread(file_img)
    img = cv2.resize(img, (640, 512))
    
    out_name = out_folder + files_names[i]
    cv2.imwrite(out_name, img)
    print(out_name)
