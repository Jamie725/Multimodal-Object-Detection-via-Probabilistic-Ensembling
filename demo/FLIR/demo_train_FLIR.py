# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import torch

def test(cfg, dataset_name):
    cfg.DATASETS.TEST = (dataset_name, )
    predictor = DefaultPredictor(cfg)
    evaluator_FLIR = FLIREvaluator(dataset_name, cfg, False, output_dir=out_folder, out_pr_name='pr_val.png')
    val_loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(predictor.model, val_loader, evaluator_FLIR)

#Set GPU
torch.cuda.set_device(0)

# get path
dataset = 'FLIR'
# Train path
train_path = '../../../Datasets/'+ dataset +'/train/'
train_folder = '../../../Datasets/FLIR/train/thermal_8_bit/'
train_json_path = '../../../Datasets/'+dataset+'/train/FLIR_thermal_RGBT_pairs_train.json'
# Validation path
val_path = '../../../Datasets/'+ dataset +'/val/'
val_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
val_json_path = '../../../Datasets/'+dataset+'/val/FLIR_thermal_RGBT_pairs_val.json'

# Register dataset
dataset_train = 'FLIR_train'
register_coco_instances(dataset_train, {}, train_json_path, train_folder)
FLIR_metadata_train = MetadataCatalog.get(dataset_train)
dataset_dicts_train = DatasetCatalog.get(dataset_train)
dataset_test = 'FLIR_val'
register_coco_instances(dataset_test, {}, val_json_path, val_folder)
FLIR_metadata_test = MetadataCatalog.get(dataset_test)
dataset_dicts_test = DatasetCatalog.get(dataset_test)

out_folder = 'temp'
out_model_path = os.path.join(out_folder, 'out_model_middle_fusion.pth')
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
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

###### Performance tuning ########
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000
eval_every_iter = 1000
cfg.MODEL.ROI_HEADS.ENABLE_GAUSSIANNLLOSS = True

########################################################
# Method
# Choice: thermal_only, early_fusion, middle_fusion
########################################################
method = 'early_fusion'

if method == "thermal_only":
    cfg.MODEL.WEIGHTS = "trained_models/Detectron2_pretrained/model_final_f6e8b1.pkl"
elif method == "early_fusion":
    cfg.INPUT.FORMAT = 'BGRT'
    cfg.INPUT.NUM_IN_CHANNELS = 4
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth'
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
elif method == "middle_fusion":
    cfg.INPUT.FORMAT = 'BGRTTT'
    cfg.INPUT.NUM_IN_CHANNELS = 6
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth'
    cfg.MODEL.BACKBONE.FREEZE_AT = 3

num_loops = cfg.SOLVER.MAX_ITER // eval_every_iter
cfg.SOLVER.MAX_ITER = eval_every_iter

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
best_AP50 = 0

for idx in range(num_loops):
    print('##############################################################################')
    print('============== The ', idx+1, ' * ', eval_every_iter, ' iterations ============')
    print('##############################################################################')    
    
    if idx > 0:
        cfg.MODEL.WEIGHTS = out_model_path
        trainer = DefaultTrainer(cfg)        
        trainer.resume_or_load(resume=False)
        out_name = 'out_model_middle_fusion.pth'
        out_model_path = os.path.join(out_folder, out_name)
    
    trainer.train()
    
    # Save model
    out_model_path = os.path.join(out_folder, 'out_model_middle_fusion.pth')
    torch.save(trainer.model.state_dict(), out_model_path)
    cfg.MODEL.WEIGHTS = out_model_path
    # Evaluation    
    results = test(cfg, dataset_test)
    AP50 = results['bbox']['AP50']
    if AP50 > best_AP50:
        best_AP50 = AP50
        out_model_path = os.path.join(out_folder, 'out_model_middle_fusion_best.pth')
        torch.save(trainer.model.state_dict(), out_model_path)
    del trainer    