# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from tools.plain_train_net import do_test

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import torch


def test(cfg, dataset_name):
    
    cfg.DATASETS.TEST = (dataset_name, )
    predictor = DefaultPredictor(cfg)
    evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, out_pr_name='pr_val.png')
    #DefaultTrainer.test(cfg, trainer.model, evaluators=evaluator_FLIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator_FLIR)

#Set GPU
torch.cuda.set_device(1)

# get path
dataset = 'FLIR'
# Train path
train_folder = '../../../Datasets/FLIR/train/RGB'
#train_folder = '../../../Datasets/FLIR/train/thermal_8_bit'
#train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4class.json'
train_json_path = '../../../Datasets/'+dataset+'/train/RGB_annotations_4_channel_no_dogs.json'
# Validation path
#val_folder = '../../../Datasets/FLIR/val/thermal_8_bit'
val_folder = '../../../Datasets/FLIR/val/RGB'
#val_json_path = '../../../Datasets/'+dataset+'/val/thermal_annotations_4class.json'
val_json_path = '../../../Datasets/'+dataset+'/val/RGB_annotations_4_channel_no_dogs.json'
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
out_folder = 'output_val'
out_model_path = os.path.join(out_folder, 'out_model_final.pth')
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# Create config
cfg = get_cfg()
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/FLIR-Detection/faster_rcnn_R_101_FLIR.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.INPUT.NUM_IN_CHANNELS = 3
#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
#cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
# Train config
cfg.DATASETS.TRAIN = (dataset_train,)
cfg.DATASETS.TEST = (dataset_test, )
#cfg.TEST.EVAL_PERIOD = 50
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

eval_every_iter = 1000
num_loops = cfg.SOLVER.MAX_ITER // eval_every_iter
cfg.SOLVER.MAX_ITER = eval_every_iter
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

for idx in range(num_loops):
    print('============== The ', idx, ' * ', eval_every_iter, ' iterations ============')    
    
    if idx > 0:
        cfg.MODEL.WEIGHTS = out_model_path
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
    
        out_name = 'out_model_iter_'+ str(idx*eval_every_iter) +'.pth'
        out_model_path = os.path.join(out_folder, out_name)
    trainer.train()
    torch.save(trainer.model.state_dict(), out_model_path)
    #pdb.set_trace()

    test(cfg, dataset_test)

    del trainer


# Test on training set
cfg.DATASETS.TEST = (dataset_train, )
predictor = DefaultPredictor(cfg)
evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='FLIR_train_eval.out')
val_loader = build_detection_test_loader(cfg, dataset_train)
inference_on_dataset(predictor.model, val_loader, evaluator)

# Test on evaluation set
cfg.DATASETS.TEST = (dataset_test, )
predictor = DefaultPredictor(cfg)
evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='FLIR_train_eval.out')
val_loader = build_detection_test_loader(cfg, dataset_test)
inference_on_dataset(predictor.model, val_loader, evaluator)
