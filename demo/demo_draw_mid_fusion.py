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

# get path
#mypath = 'input/FLIR/Day/'
dataset = 'FLIR'
train_or_val = 'val'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/RGB/'

files_names = [f for f in listdir(path) if isfile(join(path, f))]
out_folder = 'out/middle_fusion_image_result'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
# -------- Setting for 6 inputs -------- #
cfg.INPUT.FORMAT = 'BGRTTT'
cfg.INPUT.NUM_IN_CHANNELS = 6 #4
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cfg.MODEL.WEIGHTS = "good_model/mid_fusion/out_model_iter_42000.pth"
# -------------------------------------- #
#Draw trained thermal
#cfg.MODEL.WEIGHTS = os.path.join('output_val/good_model', "model_0009999.pth")
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

# Create predictor

predictor = DefaultPredictor(cfg)
for i in range(len(files_names)):
    # ------------ Prepare for inputs ------------- #
    files_names[i] = 'FLIR_09365.jpg'
    path_t = '../../../Datasets/'+ dataset +'/'+train_or_val+'/thermal_8_bit/'
    file_img = path_t + files_names[i].split(".")[0] + '.jpeg'
    thermal_img = cv2.imread(file_img)
    
    file_RGB = path + files_names[i]
    if os.path.isfile(file_RGB):
        rgb_img = cv2.imread(file_RGB)
        rgb_img = cv2.resize(rgb_img,(thermal_img.shape[1], thermal_img.shape[0]))
        image = np.zeros((thermal_img.shape[0], thermal_img.shape[1], 6))
        image [:,:,0:3] = rgb_img
        image [:,:,3:6] = thermal_img
        # ---------------------------------- #
        #model = 'faster_rcnn_R_101_FPN_3x'
        print('file = ',file_img)
        
        # Make prediction
        outputs = predictor(image)
        
        name = files_names[i].split('.')[0] + '.jpg'
        #print('name = ', files_names[i])
        out_name = out_folder +'/'+ name
        #out_name = 'FLIR_08743_thermal.jpg'
        print(out_name)

        v = Visualizer(thermal_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow('img_t',v.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        v.save(out_name)
        #pdb.set_trace()
        