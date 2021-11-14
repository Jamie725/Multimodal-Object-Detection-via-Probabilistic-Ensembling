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

def test(cfg, dataset_name, file_name='FLIR_thermal_only_result.out'):
    
    cfg.DATASETS.TEST = (dataset_name, )
    predictor = DefaultPredictor(cfg)
    #evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, out_pr_name='pr_val.png')
    
    out_name = out_folder + file_name
    #pdb.set_trace()
    evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path=out_name)
    #DefaultTrainer.test(cfg, trainer.model, evaluators=evaluator_FLIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator_FLIR)
#Set GPU
torch.cuda.set_device(1)

# get path
dataset = 'KAIST'

# Validation path
val_folder = '../../../Datasets/' + dataset + '/test/'
#val_folder = '../../Others/flownet2/flownet2-pytorch/KAIST_img/KAIST_test_flow/'
#val_json_path = '../../../Datasets/'+dataset+'/val/thermal_annotations_4class.json'
val_json_path = '../../../Datasets/'+dataset+'/test/KAIST_test_RGB_annotation.json'

print('test json path:', val_json_path)

model = 'faster_rcnn_R_101_FPN_3x'

out_folder = '.'

# Create config
cfg = get_cfg()
cfg.DATALOADER.NUM_WORKERS = 6
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.WEIGHTS = 'good_model/KAIST/RGB/out_model_iter_2500.pth'

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = (dataset, )

### 3 Channel input ###
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3
cfg.MODEL.PIXEL_MEAN = [225.328, 226.723, 235.070]#[103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
#######################

# Register dataset
# Test on validation set
dataset_test = 'KAIST_test'
register_coco_instances(dataset_test, {}, val_json_path, val_folder)
FLIR_metadata_test = MetadataCatalog.get(dataset_test)
dataset_dicts_test = DatasetCatalog.get(dataset_test)

file_name = 'FLIR_thermal_only_3_class.out'
test(cfg, dataset_test, file_name)