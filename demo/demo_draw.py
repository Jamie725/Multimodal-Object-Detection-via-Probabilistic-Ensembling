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
dataset = 'FLIR'
path = '../../../Datasets/'+ dataset +'/train/resized_RGB/'

files_names = [f for f in listdir(path) if isfile(join(path, f))]
out_folder = 'out/resized_rgb_detection_result'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Draw RGB
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

#Draw trained thermal
#cfg.MODEL.WEIGHTS = os.path.join('output_val/good_model', "model_0009999.pth")
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

# Create predictor
predictor = DefaultPredictor(cfg)

for i in range(len(files_names)):
    # get image
    
    path_t = '../../../Datasets/'+ dataset +'/train/thermal_8_bit/'
    file_img = path_t + files_names[i].split(".")[0] + '.jpeg'
    img_t = cv2.imread(file_img)

    #file_RGB = path + files_names[i]
    file_RGB = path + 'FLIR_08473.jpg'
    img_rgb = cv2.imread(file_RGB)
    
    #model = 'faster_rcnn_R_101_FPN_3x'
    print('file = ',file_img)
    
    # Make prediction
    outputs = predictor(img_rgb)
    
    name = files_names[i].split('.')[0] + '.jpg'
    #print('name = ', files_names[i])
    out_name = out_folder +'/'+ name
    #out_name = 'FLIR_08743_thermal.jpg'
    print(out_name)

    v = Visualizer(img_t[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('img_t',v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    v.save(out_name)
    pdb.set_trace()
    