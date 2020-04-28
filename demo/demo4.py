# import some common detectron2 utilities
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import pickle

# get path
#mypath = 'input/FLIR/Day/'
dataset = 'FLIR'
#time = 'Day'
#usage = 'train'
img_folder = '../../../Datasets/FLIR/train/thermal_8_bit'
json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_new2.json'

print(json_path)
#List every files in the folder
files_names = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]

# Register dataset
register_coco_instances(dataset, {}, json_path, img_folder)

FLIR_metadata = MetadataCatalog.get("FLIR")
dataset_dicts = DatasetCatalog.get("FLIR")

out_folder = 'output_' + dataset

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

model = 'faster_rcnn_R_101_FPN_3x'

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("FLIR",)
cfg.DATASETS.TEST = ("FLIR")
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # only has one class (ballon)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
#trainer.train()
"""
# Create predictor
predictor = DefaultPredictor(cfg)

for i in range(len(files_names)):
#for i in range(1000):
    # get image
    
    file_thermal = img_folder + '/' + files_names[i]
    img_thermal = cv2.imread(file_thermal)
    print('name = ',file_thermal)
    
    
    # Make prediction
    #pdb.set_trace()
    outputs = predictor(img_thermal)

    name = files_names[i].split('.')[0]
    out_name = out_folder +'/'+ name + '_' + model + '_result.jpg'

    v = Visualizer(img_thermal[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('img',v.get_image()[:, :, ::-1])
    v.save(out_name)
    #cv2.waitKey()
    #pdb.set_trace()

"""

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("FLIR", )
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from tools.plain_train_net import do_test

#pdb.set_trace()
evaluator = COCOEvaluator("FLIR", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "FLIR")

inference_on_dataset(trainer.model, val_loader, evaluator)

#pdb.set_trace()
