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

def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    pdb.set_trace()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
		utils.transform_instance_annotations(obj, transforms, image.shape[:2])
		for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0
	]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

def test(cfg, dataset_name):
    
    cfg.DATASETS.TEST = (dataset_name, )
    predictor = DefaultPredictor(cfg)
    #evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, out_pr_name='pr_val.png')
    out_name = out_folder+'FLIR_thermal_only_result.out'
    evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path=out_name)
    #DefaultTrainer.test(cfg, trainer.model, evaluators=evaluator_FLIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator_FLIR)
#Set GPU
torch.cuda.set_device(1)

# get path
dataset = 'FLIR'
out_folder = 'out/mAP/'

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
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.WEIGHTS = "good_model/thermal_only/model_0009999.pth"

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
cfg.DATASETS.TEST = (dataset, )
"""
### 4 Channel input ###
cfg.INPUT.FORMAT = 'BGRT'
cfg.INPUT.NUM_IN_CHANNELS = 4
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
#######################
"""
### 3 Channel input ###
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
#######################


# Test on validation set
dataset_test = 'FLIR_val'
register_coco_instances(dataset_test, {}, val_json_path, val_folder)
FLIR_metadata_test = MetadataCatalog.get(dataset_test)
dataset_dicts_test = DatasetCatalog.get(dataset_test)
test(cfg, dataset_test)