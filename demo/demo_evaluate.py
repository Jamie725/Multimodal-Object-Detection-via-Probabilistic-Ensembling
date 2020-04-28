# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import torch

#Set GPU
torch.cuda.set_device(1)

# get path
dataset = 'FLIR'

val_path = '../../../Datasets/'+ dataset +'/val/thermal_8_bit/'
train_folder = '../../../Datasets/' + dataset + '/train/thermal_8_bit'
val_folder = '../../../Datasets/' + dataset + '/val/thermal_8_bit'
val_json_path = '../../../Datasets/'+dataset+'/val/thermal_annotations_new.json'

out_folder = 'output_val'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# Create config
cfg = get_cfg()
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "output_val/model_0009999.pth"

# Train config
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = (dataset, )
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from tools.plain_train_net import do_test

# Test on validation set
dataset = 'FLIR_val'
cfg.DATASETS.TEST = (dataset, )
register_coco_instances(dataset, {}, val_json_path, val_folder)
FLIR_metadata = MetadataCatalog.get(dataset)
dataset_dicts = DatasetCatalog.get(dataset)

evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder)
val_loader = build_detection_test_loader(cfg, dataset)
inference_on_dataset(predictor.model, val_loader, evaluator)