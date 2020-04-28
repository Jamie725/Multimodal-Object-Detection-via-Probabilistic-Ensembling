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
dataset = 'KAIST'
mode = 'train'
set_num = '00'
img_folder = '../../../Datasets/'+ dataset +'/'+ mode + '/set' + set_num + '/V000/visible'

files_names = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]

out_folder = 'output_' + dataset
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
model = 'faster_rcnn_R_101_FPN_3x'

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
#cfg.SOLVER.IMS_PER_BATCH = 2
#cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
#trainer.train()

# Create predictor
predictor = DefaultPredictor(cfg)
for i in range(len(files_names)):
    # get image
    
    file_thermal = img_folder + '/' + files_names[i]
    img_thermal = cv2.imread(file_thermal)
    print('name = ',file_thermal)
    
    # Make prediction
    
    outputs = predictor(img_thermal)
    pdb.set_trace()

    name = files_names[i].split('.')[0]
    out_name = out_folder +'/'+ name + '_' + model + '_result.jpg'

    v = Visualizer(img_thermal[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('img',v.get_image()[:, :, ::-1])
    pdb.set_trace()
    v.save(out_name)