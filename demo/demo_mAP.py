# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from os.path import isfile, join
import numpy as np
import torch
import cv2
import os
import pickle
import pdb

#Set GPU
torch.cuda.set_device(1)

# get path
dataset = 'FLIR'
out_folder = 'output_4_channel'
"""
dataset_part = 'val'
json_name = 'thermal_annotations_new.json'

path = '../../../Datasets/'+ dataset +'/'+dataset_part+'/thermal_8_bit/'
img_folder = '../../../Datasets/'+ dataset +'/'+dataset_part+'/thermal_8_bit'
json_path = '../../../Datasets/'+dataset+'/'+dataset_part+'/'+json_name
print(json_path)
"""
# Train path
train_path = '../../../Datasets/'+ dataset +'/train/thermal_8_bit/'
train_folder = '../../../Datasets/FLIR/train/thermal_8_bit'
#train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_small.json'
train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4_channel_no_dogs.json'
# Validation path
val_path = '../../../Datasets/'+ dataset +'/val/thermal_8_bit/'
val_folder = '../../../Datasets/FLIR/val/thermal_8_bit'
#val_json_path = '../../../Datasets/'+dataset+'/val/thermal_annotations_new.json'
val_json_path = '../../../Datasets/'+dataset+'/val/thermal_annotations_4_channel_no_dogs.json'

# Register dataset
dataset = 'FLIR_train'
register_coco_instances(dataset, {}, train_json_path, train_folder)
FLIR_metadata = MetadataCatalog.get(dataset)
dataset_dicts = DatasetCatalog.get(dataset)

model = 'faster_rcnn_R_101_FPN_3x'

# Create config
cfg = get_cfg()
cfg.DATALOADER.NUM_WORKERS = 6
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "good_model/out_model_iter_32000.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
cfg.DATASETS.TEST = (dataset, )
"""
### 4 Channel input ###
cfg.INPUT.FORMAT = 'BGRT'
cfg.INPUT.NUM_IN_CHANNELS = 4
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
#######################
"""
### 4 Channel input ###
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#######################

predictor = DefaultPredictor(cfg)
"""
# Training data
evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='FLIR_noT_train_eval.out')
val_loader = build_detection_test_loader(cfg, dataset)
inference_on_dataset(predictor.model, val_loader, evaluator)
"""
# Test on validation set
dataset = 'FLIR_val'
cfg.DATASETS.TEST = (dataset, )
register_coco_instances(dataset, {}, val_json_path, val_folder)
FLIR_metadata = MetadataCatalog.get(dataset)
dataset_dicts = DatasetCatalog.get(dataset)

# Validation data
evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='FLIR_noT_val_eval.out')
val_loader = build_detection_test_loader(cfg, dataset)
inference_on_dataset(predictor.model, val_loader, evaluator)

