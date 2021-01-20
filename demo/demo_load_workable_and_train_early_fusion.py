# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from tools.plain_train_net import do_test
from detectron2.config import CfgNode as CN
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import torch
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

def test_during_train(trainer, dataset_name):
    cfg.DATASETS.TEST = (dataset_name, )
    evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, out_pr_name='pr_val.png')
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(trainer.model, val_loader, evaluator_FLIR)

def test(cfg, dataset_name):
    
    cfg.DATASETS.TEST = (dataset_name, )
    predictor = DefaultPredictor(cfg)
    evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, out_pr_name='pr_val.png')
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator_FLIR)

# Set GPU
torch.cuda.set_device(1)
# [1] 20599

# get path
dataset = 'FLIR'
# Train path
train_path = '../../../Datasets/'+ dataset +'/train/thermal_8_bit/'
train_folder = '../../../Datasets/FLIR/train/thermal_8_bit'
train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4_channel_no_dogs_3_class.json'

# Validation path
val_path = '../../../Datasets/'+ dataset +'/val/thermal_8_bit/'
val_folder = '../../../Datasets/FLIR/val/thermal_8_bit'
val_json_path = '../../../Datasets/'+dataset+'/val/thermal_annotations_4_channel_no_dogs_3_class.json'
print(train_json_path)

# Register dataset
dataset_train = 'FLIR_train'
register_coco_instances(dataset_train, {}, train_json_path, train_folder)
FLIR_metadata_train = MetadataCatalog.get(dataset_train)
dataset_dicts_train = DatasetCatalog.get(dataset_train)

# Test on validation set
dataset_test = 'FLIR_val'
register_coco_instances(dataset_test, {}, val_json_path, val_folder)
FLIR_metadata_test = MetadataCatalog.get(dataset_test)
dataset_dicts_test = DatasetCatalog.get(dataset_test)

model = 'faster_rcnn_R_101_FPN_3x'

#files_names = [f for f in listdir(train_path) if isfile(join(train_path, f))]

out_folder = 'output_early_fusion_3_class_load_workable_freeze_most_param_0106_2'

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# Create config
cfg = get_cfg()
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/FLIR-Detection/faster_rcnn_R_101_FLIR.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Train config
cfg.DATASETS.TRAIN = (dataset_train,)
cfg.DATASETS.TEST = (dataset_test, )
#cfg.TEST.EVAL_PERIOD = 50
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

###### Performance tuning ########
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

# Pid = 4830 -> gpu0
# Pid = 17314-> gpu1

###### Parameter for 3 channel input ####

#cfg.MODEL.WEIGHTS = 'good_model/3_class/thermal_only/out_model_iter_15000.pth' # thermal only
cfg.MODEL.WEIGHTS = 'good_model/early_fusion//out_model_iter_12000.pth' # thermal only
cfg.INPUT.FORMAT = 'BGRT'
cfg.INPUT.NUM_IN_CHANNELS = 4
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
#cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#########################################

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Loading parameters of the first layer and concatenate them to 4 channels
checkpoint = torch.load(cfg.MODEL.WEIGHTS)
from detectron2.modeling import build_model

rm_name_list = ['roi_heads.box_predictor.cls_score.weight', 'roi_heads.box_predictor.cls_score.bias', 'roi_heads.box_predictor.bbox_pred.weight', 'roi_heads.box_predictor.bbox_pred.bias']
pretrained_dict = {}
for name, param in checkpoint.items():
    if name in rm_name_list:
        if 'cls_score.weight' in name:
            param_new = torch.zeros((4, param.shape[1]))
            param_new[:3,:] = param[:3,:]
            param_new[-1,:] = param[-1,:]
            pretrained_dict[name] = param_new
        elif 'cls_score.bias' in name:
            param_new = torch.zeros(4)
            param_new[:3] = param[:3]
            param_new[-1] = param[-1]
            pretrained_dict[name] = param_new
        elif 'bbox_pred.weight' in name:
            param_new = torch.zeros((12, param.shape[1]))
            param_new[:9,:] = param[:9,:]
            param_new[-4:,:] = param[-4:,:]
            pretrained_dict[name] = param_new
        elif 'bbox_pred.bias' in name:
            param_new = torch.zeros(12)
            param_new[:9] = param[:9]
            param_new[-4:] = param[-4:]
            pretrained_dict[name] = param_new
    else:
        pretrained_dict[name] = param
del checkpoint
#model_final.load_state_dict(pretrained_dict)

# Set for training 4 inputs
cfg.INPUT.FORMAT = 'BGRT'
cfg.INPUT.NUM_IN_CHANNELS = 4
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
#cfg.MODEL.BACKBONE.FREEZE_AT = 0
#cfg.MODEL.WEIGHTS = 'good_model/3_class/thermal_only/out_model_iter_15000.pth'
#cfg.INPUT.CROP = CN({"ENABLED": True})
eval_every_iter = 100
num_loops = cfg.SOLVER.MAX_ITER // eval_every_iter
cfg.SOLVER.MAX_ITER = eval_every_iter
out_model_path = os.path.join(out_folder, 'out_model_iter_'+ str(eval_every_iter) +'.pth')
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
cnt = 0

# Load pretrained model parameters
trainer.model.load_state_dict(pretrained_dict)
del pretrained_dict

"""
for p in trainer.model.parameters():
    p.requires_grad = False
    
for p in trainer.model.roi_heads.parameters():
    p.requires_grad = True
"""
for name, param in trainer.model.named_parameters():
    if name in rm_name_list:
        param.requires_grad = True
    else:
        param.requires_grad = False

for idx in range(1,num_loops):
    print('============== The ', idx, ' * ', eval_every_iter, ' iterations ============')    
    
    if idx > 1:
        cfg.MODEL.WEIGHTS = out_model_path
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        
        for name, param in trainer.model.named_parameters():
            if name in rm_name_list:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        out_name = 'out_model_iter_'+ str(idx*eval_every_iter) +'.pth'
        out_model_path = os.path.join(out_folder, out_name)
    
    trainer.train()
    torch.save(trainer.model.state_dict(), out_model_path)

    # Evaluation on validation set
    test_during_train(trainer, dataset_train)
    test_during_train(trainer, dataset_test)
    del trainer
