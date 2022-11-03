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

# get path
dataset = 'FLIR'
ther_path = '../../../Datasets/'+ dataset +'/val/thermal_8_bit/'
rgb_path = '../../../Datasets/'+ dataset +'/val/RGB/'

files_names = [f for f in listdir(rgb_path) if isfile(join(rgb_path, f))]
out_folder = 'tmp/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

########################################################
# Method
# Choice: thermal_only, early_fusion, middle_fusion
########################################################
method = 'mid_fusion'
if method == "thermal_only":
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth'    
elif method == "early_fusion":
    cfg.INPUT.FORMAT = 'BGRT'
    cfg.INPUT.NUM_IN_CHANNELS = 4
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/early_fusion/out_model_early_fusion.pth'
elif method == "mid_fusion":
    cfg.INPUT.FORMAT = 'BGRTTT'
    cfg.INPUT.NUM_IN_CHANNELS = 6
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/mid_fusion/out_model_middle_fusion.pth'    

# Create predictor
predictor = DefaultPredictor(cfg)

for i in range(len(files_names)):
    file_name_rgb = rgb_path + files_names[i]
    file_name_ther = ther_path + files_names[i].split('.')[0] + '.jpeg'
    rgb_img = cv2.imread(file_name_rgb)
    ther_img = cv2.imread(file_name_ther)    
    ther_img = cv2.resize(ther_img, (rgb_img.shape[1], rgb_img.shape[0]))
    if method == "thermal_only":        
        img = ther_img
    elif method == "early_fusion":        
        img = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 4))
        img[:,:,0:3] = rgb_img        
        img[:,:,3] = ther_img[:,:,0]
    elif method == "mid_fusion":
        img = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 6))
        img[:,:,0:3] = rgb_img        
        img[:,:,3:] = ther_img
    print('file = ',file_name_rgb)
  
    # Make prediction
    outputs = predictor(img)
    name = files_names[i].split('.')[0] + '_thermal.jpg'
    out_name = out_folder +'/'+ name
    print(out_name)

    # Overlay the detected boxes on thermal image
    v = Visualizer(ther_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v.save(out_name)

    # Overlay the detected boxes on RGB image
    name = files_names[i].split('.')[0] + '_rgb.jpg'
    out_name = out_folder +'/'+ name
    v = Visualizer(rgb_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v.save(out_name)