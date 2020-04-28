# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
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
path = '../../../Datasets/'+ dataset +'/train/thermal_8_bit/'
img_folder = '../../../Datasets/FLIR/train/thermal_8_bit'
json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_new2.json'
print(json_path)

# Register dataset
register_coco_instances(dataset, {}, json_path, img_folder)
FLIR_metadata = MetadataCatalog.get("FLIR")
dataset_dicts = DatasetCatalog.get("FLIR")
model = 'faster_rcnn_R_101_FPN_3x'

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

files_names = [f for f in listdir(path) if isfile(join(path, f))]

out_folder = 'output_6'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# Create config
cfg = get_cfg()
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

# Train config
cfg.DATASETS.TRAIN = ("FLIR",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("FLIR", )
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from tools.plain_train_net import do_test

#pdb.set_trace()
evaluator = FLIREvaluator("FLIR", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "FLIR")
#print('here!')
inference_on_dataset(predictor.model, val_loader, evaluator)
#print('here!')