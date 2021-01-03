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
#mypath = 'input/FLIR/Day/'
dataset = 'FLIR'
train_or_val = 'val'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/RGB/'

files_names = [f for f in listdir(path) if isfile(join(path, f))]
out_folder = 'out/img/thermal_only_results_paper'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

torch.cuda.set_device(1)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Draw RGB
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
#cfg.MODEL.WEIGHTS = "good_model/mid_fusion/out_model_iter_42000.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
# -------- Setting for 6 inputs -------- #
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3 #4
#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#cfg.MODEL.WEIGHTS = "good_model/thermal_only/model_0009999.pth"
# -------------------------------------- #
#Draw trained thermal
cfg.MODEL.WEIGHTS = os.path.join('output_val/good_model', "model_0009999.pth")
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

# Create predictor
predictor = DefaultPredictor(cfg)

for i in range(len(files_names)):
    # get image
    files_names[i] = 'FLIR_09436.jpg'
    path_t = '../../../Datasets/'+ dataset +'/'+train_or_val+'/thermal_8_bit/'
    file_img = path_t + files_names[i].split(".")[0] + '.jpeg'
    img_t = cv2.imread(file_img)

    file_RGB = path + files_names[i]
    img_rgb = cv2.imread(file_RGB)
    
    #model = 'faster_rcnn_R_101_FPN_3x'
    print('file = ',file_img)
    
    # Make prediction
    outputs = predictor(img_t)
    name = files_names[i].split('.')[0] + '_thermal_only.jpg'
    #print('name = ', files_names[i])
    out_name = out_folder +'/'+ name
    print(outputs)
    pdb.set_trace()
    num_box = len(outputs['instances']._fields['pred_boxes'])
    for j in range(num_box):
        bbox = outputs['instances']._fields['pred_boxes'][j].tensor.cpu().numpy()
        img_t = cv2.rectangle(img_t, (bbox[0][0], bbox[0][1]), (bbox[0][2],bbox[0][3]), (0,255,0), 2)
    print(out_name)
    
    cv2.imwrite(out_name, img_t)

    #v = Visualizer(img_t[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('img_t',v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    #v.save(out_name)
    #pdb.set_trace()