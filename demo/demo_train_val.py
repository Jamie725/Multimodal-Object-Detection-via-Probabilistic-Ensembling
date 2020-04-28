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
import pdb

#Set GPU
torch.cuda.set_device(1)

# get path
dataset = 'FLIR'
# Train path
train_path = '../../../Datasets/'+ dataset +'/train/thermal_8_bit/'
train_folder = '../../../Datasets/FLIR/train/thermal_8_bit'
train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_new2.json'
# Validation path
val_path = '../../../Datasets/'+ dataset +'/val/thermal_8_bit/'
val_folder = '../../../Datasets/FLIR/val/thermal_8_bit'
val_json_path = '../../../Datasets/'+dataset+'/val/thermal_annotations_new.json'
print(train_json_path)

# Register dataset
dataset = 'FLIR_train'
register_coco_instances(dataset, {}, train_json_path, train_folder)
FLIR_metadata = MetadataCatalog.get(dataset)
dataset_dicts = DatasetCatalog.get(dataset)

model = 'faster_rcnn_R_101_FPN_3x'

files_names = [f for f in listdir(train_path) if isfile(join(train_path, f))]

out_folder = 'output_val'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# Create config
cfg = get_cfg()
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

# Train config
cfg.DATASETS.TRAIN = (dataset,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from tools.plain_train_net import do_test

# Test on training set
cfg.DATASETS.TEST = (dataset, )
predictor = DefaultPredictor(cfg)
evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, out_pr_name='pr_train.png')
val_loader = build_detection_test_loader(cfg, dataset)
inference_on_dataset(predictor.model, val_loader, evaluator)

# Test on validation set
dataset = 'FLIR_val'
cfg.DATASETS.TEST = (dataset, )
register_coco_instances(dataset, {}, val_json_path, val_folder)
FLIR_metadata = MetadataCatalog.get(dataset)
dataset_dicts = DatasetCatalog.get(dataset)

evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, out_pr_name='pr_val.png')
val_loader = build_detection_test_loader(cfg, dataset)
inference_on_dataset(predictor.model, val_loader, evaluator)
