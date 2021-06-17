# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer_paper import Visualizer
from detectron2.data import MetadataCatalog
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import torch

# get path
dataset = 'FLIR'
train_or_val = 'val'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/thermal_8_bit/'
files_names = [f for f in listdir(path) if isfile(join(path, f))]
out_folder = 'out/img/thermal_only/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

torch.cuda.set_device(0)
# -------------------------------------- #
cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
# -------------------------------------- #
#Draw trained thermal
cfg.MODEL.WEIGHTS = os.path.join('good_model/thermal_only', "out_model_iter_15000.pth")

# Create predictor
predictor = DefaultPredictor(cfg)

for i in range(len(files_names)):
    # get image    
    path_t = '../../../Datasets/'+ dataset +'/'+train_or_val+'/thermal_8_bit/'
    file_img = path_t + files_names[i].split(".")[0] + '.jpeg'
    img_t = cv2.imread(file_img)
    file_RGB = path + files_names[i]
    img_rgb = cv2.imread(file_RGB)   
    print('file = ',file_img)
    
    # Make prediction
    outputs = predictor(img_t)
    name = files_names[i].split('.')[0] + '.jpg'
    out_name = out_folder +'/'+ name
    print(out_name)

    v = Visualizer(img_t[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v.save(out_name)
