# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import torch

def test(cfg, dataset_name, save_eval_name, save_folder):
    cfg.DATASETS.TEST = (dataset_name, )
    predictor = DefaultPredictor(cfg)
    evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=save_folder, save_eval=True, out_eval_path=(save_folder + save_eval_name))    
    val_loader = build_detection_test_loader(cfg, dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator_FLIR)

out_folder = './'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

#Set GPU
torch.cuda.set_device(0)
### Dataset setting ###
dataset = 'FLIR'
# Validation path
val_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
val_json_path = '../../../Datasets/'+dataset+'/val/FLIR_thermal_RGBT_pairs_val.json'#FLIR_thermal_RGBT_pairs_val.json'
# Test on validation set
dataset_test = 'FLIR_val'
register_coco_instances(dataset_test, {}, val_json_path, val_folder)
FLIR_metadata_test = MetadataCatalog.get(dataset_test)
dataset_dicts_test = DatasetCatalog.get(dataset_test)

# Create config
cfg = get_cfg()
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/FLIR-Detection/faster_rcnn_R_101_FLIR.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.DATASETS.TEST = (dataset_test, )
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

########################################################
# Method
# Choice: thermal_only, early_fusion, middle_fusion
########################################################
method = 'thermal_only'
if method == "thermal_only":
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth'
    output_file = 'FLIR_thermal_only_mAP.out'
elif method == "early_fusion":
    cfg.INPUT.FORMAT = 'BGRT'
    cfg.INPUT.NUM_IN_CHANNELS = 4
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/early_fusion/out_model_early_fusion.pth'
    output_file = 'FLIR_early_fusion_mAP.out'
elif method == "middle_fusion":
    cfg.INPUT.FORMAT = 'BGRTTT'
    cfg.INPUT.NUM_IN_CHANNELS = 6
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/middle_fusion/out_model_middle_fusion.pth'
    output_file = 'FLIR_middle_fusion_mAP.out'

test(cfg, dataset_test, output_file, out_folder)