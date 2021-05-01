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
train_or_val = 'video'
img_type = 'thermal_8_bit'#'RGB'#'thermal_8_bit'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/'+img_type+'/'

files_names = [f for f in listdir(path) if isfile(join(path, f))]
if img_type == 'RGB':
    out_folder = 'iccv/RGB_0_5_text/'
    thr = 0.5
else:
    out_folder = 'iccv/thermal_0_8_text/'
    thr = 0.8
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

torch.cuda.set_device(1)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
if img_type == 'RGB':
    # Draw RGB
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    #cfg.MODEL.WEIGHTS = "good_model/mid_fusion/out_model_iter_42000.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
else:
    cfg.MODEL.WEIGHTS = "good_model/3_class/thermal_only/out_model_iter_15000.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.PIXEL_MEAN = [135.438, 135.438, 135.438]#Thermal
# General settings
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3 #4
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

# Create predictor
predictor = DefaultPredictor(cfg)
class_name = ['person', 'bicycle', 'car']
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 0.7
# Line thickness of 2 px
thickness = 2
color = (0, 130, 255)

des_shape = np.array([512,640])

#for i in range(len(files_names)):
for i in range(1, 620):    
    in_path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/'+img_type+'/'
    if img_type == 'RGB':        
        out_name = out_folder + 'FLIR_video_' + format(i, '05d') + '_RGB.jpg'
        file_img = in_path + 'FLIR_video_' + format(i, '05d') + '.jpg'
        img = cv2.imread(file_img)
        img = cv2.resize(img, (640,512))        
    else:
        out_name = out_folder +'/'+ 'FLIR_video_' + format(i, '05d') + '_thermal_only.jpg'
        file_img = in_path + 'FLIR_video_' + format(i, '05d') + '.jpeg'
        img = cv2.imread(file_img)
    print('file = ',file_img)
    
    # Make prediction
    try:
        outputs = predictor(img)
    except:
        pdb.set_trace()
    num_box = len(outputs['instances']._fields['pred_boxes'])     
    for j in range(num_box):        
        score = outputs['instances'].scores[j].cpu().numpy()
        pred_class = outputs['instances'].pred_classes[j]        
        
        if score < thr or pred_class > 2:        
            continue
        bbox = outputs['instances']._fields['pred_boxes'][j].tensor.cpu().numpy() 
        thickness = 2
        img = cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[0][2],bbox[0][3]), (0,255,0), thickness)        
        
        thickness = 2
        min_x = max(int(bbox[0][0] - 5), 0)
        min_y = max(int(bbox[0][1] - 5), 0)        
        img = cv2.putText(img, class_name[pred_class], (min_x, min_y), font, fontScale, color, thickness, cv2.LINE_AA)
    print(out_name)
    cv2.imwrite(out_name, img)
    #pdb.set_trace()