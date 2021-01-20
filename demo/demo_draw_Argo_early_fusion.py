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

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
# -------- Setting for 4 inputs -------- #
cfg.INPUT.FORMAT = 'BGRT'
cfg.INPUT.NUM_IN_CHANNELS = 4
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
cfg.MODEL.WEIGHTS = 'good_model/3_class/early_fusion/out_model_iter_100.pth'
# -------------------------------------- #

# Create predictor
predictor = DefaultPredictor(cfg)

folders = ['daytime', 'nightApproach', 'nightHigh', 'nightLow', 'nightSmoke', 'sunset']
for i in range(len(folders)):
    time = folders[i]
    folder_name = '../../../Datasets/Argo/'+time+'/'

    out_folder = 'Argo_result/early_fusion/'+time
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for length in range(10, 210, 10):
        # ------------ Prepare for inputs ------------- #
        img_name = 'ARGO_AI_Confidential_'+time+'-%03dm_composite.png'%length
        img = cv2.imread(folder_name + img_name)
        
        img_rgb = img[:3100,:4092,:]
        #img_rgb = img[960:2100,1274:2700,:]
        img_ther = img[:1140,4097:,:]
        
        img_rgb = cv2.resize(img_rgb,(1426,1140))
        img_ther = cv2.resize(img_ther,(1426,1140))
        image = np.zeros((1140,1426, 4))
        image [:,:,0:3] = img_rgb
        image [:,:,3] = img_ther[:,:,0]
        # ---------------------------------- #
        
        # Make prediction
        outputs = predictor(image)
        
        name = time + str(length) + '.jpg'
        #print('name = ', files_names[i])
        out_name = out_folder +'/'+ name
        
        print(out_name)

        v = Visualizer(img_ther[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow('img_t',v.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        v.save(out_name)
        #pdb.set_trace()
        