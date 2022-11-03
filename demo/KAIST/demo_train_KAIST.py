# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import FLIREvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from evalKAIST.evaluation_script import evaluate
from os.path import isfile, join
import numpy as np
import torch
import cv2
import os
import pickle
import pdb
from tqdm import tqdm
import time

def evaluate_KAIST(cfg, out_file_name, data_gen, tqdm_en=False):
    # Init predictor
    predictor = DefaultPredictor(cfg)

    # Read KIAST file list
    img_folder = '../../../Datasets/KAIST/'
    file_path = img_folder + 'KAIST_evaluation/data/kaist-rgbt/splits/test-all-20.txt'
    with open(file_path) as f:contents = f.readlines()
    img_folder = '../../../Datasets/KAIST/test/'

    with open(out_file_name, mode='w') as f:
        if tqdm_en:
            for i in tqdm(range(len(contents)), desc='Predict Progress'):
                fpath = contents[i].split('\n')[0]
                set_num = fpath.split('/')[0]
                V_num = fpath.split('/')[1]
                img_num = fpath.split('/')[2]
                path_thermal = img_folder + set_num + '/' + V_num + '/lwir/' + img_num + '.jpg'
                path_rgb = img_folder + set_num + '/' + V_num + '/visible/' + img_num + '.jpg'
                
                if data_gen == 'thermal_only':
                    inputs = cv2.imread(path_thermal)
                elif data_gen == 'rgb_only':
                    inputs = cv2.imread(path_thermal)
                elif data_gen == 'early_fusion':
                    im_rgb = cv2.imread(path_rgb)
                    im_thermal = cv2.imread(path_thermal)            
                    height, width, _ = im_rgb.shape 
                    inputs = np.zeros((height, width, 4))
                    inputs[:,:,:3] = im_rgb
                    inputs[:,:,3] = im_thermal[:,:,0]
                elif data_gen == 'middle_fusion':
                    im_rgb = cv2.imread(path_rgb)
                    im_thermal = cv2.imread(path_thermal)            
                    height, width, _ = im_rgb.shape 
                    inputs = np.zeros((height, width, 6))
                    inputs[:,:,:3] = im_rgb
                    inputs[:,:,3:] = im_thermal
            
                # Make prediction
                outputs = predictor(inputs)
                # Output to file for evaluation
                num_box = len(outputs['instances']._fields['pred_boxes'])        
                for j in range(num_box):
                    pred_class = outputs['instances'].pred_classes[j].cpu().numpy()            
                    score = outputs['instances'].scores[j].cpu().numpy()
                    bbox = outputs['instances']._fields['pred_boxes'][j].tensor.cpu().numpy()[0]
                    bbox[2] -= bbox[0] 
                    bbox[3] -= bbox[1]                        
                    f.write(str(i+1)+',')
                    f.write(','.join(str(c) for c in bbox))
                    f.write(','+str(score))
                    f.write('\n')
        else:
            for i in range(len(contents)):
                fpath = contents[i].split('\n')[0]
                set_num = fpath.split('/')[0]
                V_num = fpath.split('/')[1]
                img_num = fpath.split('/')[2]
                path_thermal = img_folder + set_num + '/' + V_num + '/lwir/' + img_num + '.jpg'
                path_rgb = img_folder + set_num + '/' + V_num + '/visible/' + img_num + '.jpg'
                
                if data_gen == 'thermal_only':
                    inputs = cv2.imread(path_thermal)
                elif data_gen == 'rgb_only':
                    inputs = cv2.imread(path_thermal)
                elif data_gen == 'early_fusion':
                    im_rgb = cv2.imread(path_rgb)
                    im_thermal = cv2.imread(path_thermal)            
                    height, width, _ = im_rgb.shape 
                    inputs = np.zeros((height, width, 4))
                    inputs[:,:,:3] = im_rgb
                    inputs[:,:,3] = im_thermal[:,:,0]
                elif data_gen == 'middle_fusion':
                    im_rgb = cv2.imread(path_rgb)
                    im_thermal = cv2.imread(path_thermal)            
                    height, width, _ = im_rgb.shape 
                    inputs = np.zeros((height, width, 6))
                    inputs[:,:,:3] = im_rgb
                    inputs[:,:,3:] = im_thermal
            
                # Make prediction
                outputs = predictor(inputs)
                # Output to file for evaluation
                num_box = len(outputs['instances']._fields['pred_boxes'])        
                for j in range(num_box):
                    pred_class = outputs['instances'].pred_classes[j].cpu().numpy()            
                    score = outputs['instances'].scores[j].cpu().numpy()
                    bbox = outputs['instances']._fields['pred_boxes'][j].tensor.cpu().numpy()[0]
                    bbox[2] -= bbox[0] 
                    bbox[3] -= bbox[1]                        
                    f.write(str(i+1)+',')
                    f.write(','.join(str(c) for c in bbox))
                    f.write(','+str(score))
                    f.write('\n')
    f.close()
    result = evaluate('demo/evalKAIST/KAIST_annotation.json', out_file_name, 'Multispectral')
    out = {}
    out['all'] = result['all'].summarize(0)
    out['day'] = result['day'].summarize(0)
    out['night'] = result['night'].summarize(0)
    recall_all = 1 - result['all'].eval['yy'][0][-1]
    out['recall'] = recall_all
    return out

cfg = get_cfg()
#############################################
# Check carefully everytime 
#############################################
#GPU PID:  6847 - early fusion
torch.cuda.set_device(0)
# Model selection
model = 'faster_rcnn_R_50_FPN_3x'
data_gen = 'early_fusion'#'middle_fusion'#'early_fusion'#'rgb_only'#'thermal_only'
out_folder = 'out_KAIST_model/'+data_gen+'_gnll_0305'
out_file_name = out_folder+'/KAIST_'+data_gen+'_gnll.txt'
#out_folder = 'out_KAIST_model/'+data_gen+'_from_scratch'
#out_file_name = out_folder+'/KAIST_'+data_gen+'_from_scratch.txt'
eval_every_iter = 100
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
#cfg.MODEL.BACKBONE.FREEZE_AT = 3
cfg.MODEL.ROI_HEADS.ENABLE_GAUSSIANNLLOSS = True
#############################################
print('------- Method: ', data_gen, ' ---------')
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
# Register dataset
# Validation path
dataset = 'KAIST'
# Train dataset
train_folder = '../../../Datasets/' + dataset + '/train/'
train_json_path = '../../../Datasets/'+dataset+'/train/KAIST_train_RGB_annotation.json'
dataset_train = 'KAIST_train'
register_coco_instances(dataset_train, {}, train_json_path, train_folder)
KAIST_metadata_train = MetadataCatalog.get(dataset_train)
dataset_dicts_train = DatasetCatalog.get(dataset_train)

# Test dataset
val_folder = '../../../Datasets/' + dataset + '/test/'
val_json_path = '../../../Datasets/'+dataset+'/test/KAIST_test_RGB_annotation.json'
dataset_test = 'KAIST_test'
register_coco_instances(dataset_test, {}, val_json_path, val_folder)
KAIST_metadata_test = MetadataCatalog.get(dataset_test)
dataset_dicts_test = DatasetCatalog.get(dataset_test)

# Config settings
cfg.OUTPUT_DIR = out_folder
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 2


if data_gen == 'thermal_only':
    cfg.INPUT.FORMAT = 'BGR'
    cfg.INPUT.NUM_IN_CHANNELS = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]#[225.328, 226.723, 235.070]#
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/thermal_only/KAIST_model_thermal_only.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'
if data_gen == 'rgb_only':
    cfg.INPUT.FORMAT = 'BGR'
    cfg.INPUT.NUM_IN_CHANNELS = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]#[225.328, 226.723, 235.070]#
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    #cfg.MODEL.WEIGHTS = 'pretrained_model/model_final_f6e8b1.pkl'#'good_model/KAIST/rgb_only/KAIST_model_rgb_only.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'    
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/rgb_only/KAIST_model_rgb_only.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'    
elif data_gen == 'early_fusion':
    cfg.INPUT.FORMAT = 'BGRT'
    cfg.INPUT.NUM_IN_CHANNELS = 4
    cfg.MODEL.PIXEL_MEAN += [135.438]
    cfg.MODEL.PIXEL_STD += [1.00]
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/early_fusion/out_model_early_fusion_19_36.pth'#'/home/jamie/Desktop/kaist_final_models/early_fusion_model_23/model_0039999.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'
elif data_gen == 'middle_fusion':
    cfg.INPUT.FORMAT = 'BGRTTT'
    cfg.INPUT.NUM_IN_CHANNELS = 6
    cfg.MODEL.PIXEL_MEAN += [135.438, 135.438, 135.438]
    cfg.MODEL.PIXEL_STD += [1.00, 1.00, 1.00]
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/middle_fusion/KAIST_model_middle_fusion.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'

num_loops = cfg.SOLVER.MAX_ITER // eval_every_iter
cfg.SOLVER.MAX_ITER = eval_every_iter
cfg.DATASETS.TRAIN = (dataset_train,)
cfg.DATASETS.TEST = (dataset_test, )
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
out_name = 'out_model_'+data_gen+'_gnll.pth'
out_model_path = os.path.join(out_folder, out_name)
best_LAMR = 1000

#"""
for param in trainer.model.backbone.parameters():
    param.requires_grad = False
for param in trainer.model.roi_heads.parameters():
    param.requires_grad = False
if data_gen == 'middle_fusion':
    for param in trainer.model.backbone_2.parameters():
        param.requires_grad = False
#"""
for idx in range(num_loops):
    print('##############################################################################')
    print('============== The ', idx+1, ' * ', eval_every_iter, ' iterations ============')
    print('##############################################################################')    
    
    if idx > 0:
        cfg.MODEL.WEIGHTS = out_model_path
        trainer = DefaultTrainer(cfg)
        # Freeze
        #"""
        for param in trainer.model.backbone.parameters():
            param.requires_grad = False
        for param in trainer.model.roi_heads.parameters():
            param.requires_grad = False
        if data_gen == 'middle_fusion':
            for param in trainer.model.backbone_2.parameters():
                param.requires_grad = False

        #"""
        
        trainer.resume_or_load(resume=False)
        out_name = 'out_model_'+data_gen+'_gnll.pth'
        out_model_path = os.path.join(out_folder, out_name)
    
    trainer.train()
    
    # Save model
    out_name = 'out_model_'+data_gen+'_gnll.pth'
    out_model_path = os.path.join(out_folder, out_name)
    torch.save(trainer.model.state_dict(), out_model_path)
    cfg.MODEL.WEIGHTS = out_model_path
    # Evaluation    
    results = evaluate_KAIST(cfg, out_file_name, data_gen)
    
    #results = test(cfg, dataset_test)
    LAMR = results['all']
    if LAMR < best_LAMR:
        best_LAMR = LAMR
        out_model_path = os.path.join(out_folder, 'out_model_'+data_gen+'_best_gnll.pth')
        torch.save(trainer.model.state_dict(), out_model_path)
    del trainer    