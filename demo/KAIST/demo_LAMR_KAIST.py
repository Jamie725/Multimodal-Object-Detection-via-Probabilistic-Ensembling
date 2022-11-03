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
from tqdm import tqdm
import time
#############################################
# Check carefully everytime 
#############################################
#Set GPU
torch.cuda.set_device(1)
# Model selection
model = 'faster_rcnn_R_50_FPN_3x'
data_gen = 'middle_fusion'#'rgb_only'#'thermal_only'
out_folder = 'out/box_predictions/KAIST/'+data_gen+'/'
out_file_name = out_folder+'KAIST_'+data_gen+'_gnll.txt'
#############################################

# Register dataset
# Validation path
dataset = 'KAIST'
val_folder = '../../../Datasets/' + dataset + '/test/'
val_json_path = '../../../Datasets/'+dataset+'/test/KAIST_test_thermal_annotation.json'#KAIST_test_thermal_annotation.json'#KAIST_test_RGB_annotation.json'
dataset_test = 'KAIST_test'
register_coco_instances(dataset_test, {}, val_json_path, val_folder)
KAIST_metadata_test = MetadataCatalog.get(dataset_test)
dataset_dicts_test = DatasetCatalog.get(dataset_test)

# Create config
cfg = get_cfg()
cfg.DATALOADER.NUM_WORKERS = 6
cfg.OUTPUT_DIR = out_folder
#cfg.merge_from_file("./configs/COCO-Detection/" + model)
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

#cfg.MODEL.WEIGHTS = 'good_model/KAIST/thermal/out_model_iter_6200.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = (dataset_test, )
if data_gen == 'thermal_only':
    cfg.INPUT.FORMAT = 'BGR'
    cfg.INPUT.NUM_IN_CHANNELS = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]#[225.328, 226.723, 235.070]#
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/thermal_only/out_model_thermal_only_gnll_18_99.pth'#'/home/jamie/Desktop/kaist_final_models/output_thermal_67/model_0029999.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'
if data_gen == 'rgb_only':
    cfg.INPUT.FORMAT = 'BGR'
    cfg.INPUT.NUM_IN_CHANNELS = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]#[225.328, 226.723, 235.070]#
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    #cfg.MODEL.WEIGHTS = '/home/jamie/Desktop/kaist_final_models/output_RGB_59/model_0009999.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'
    #cfg.MODEL.WEIGHTS = 'out_KAIST_model/rgb_only_gnll_0304/out_model_rgb_only_best_gnll.pth'#'/home/jamie/Desktop/kaist_final_models/output_RGB_59/model_0004999.pth'#'out_KAIST_model/rgb_only_gnll_0304/out_model_rgb_only_best_gnll.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/rgb_only/out_model_rgb_only_best_gnll_18_67.pth'
elif data_gen == 'early_fusion':
    cfg.INPUT.FORMAT = 'BGRT'
    cfg.INPUT.NUM_IN_CHANNELS = 4
    cfg.MODEL.PIXEL_MEAN += [135.438]  # normalization!!
    cfg.MODEL.PIXEL_STD += [1.00]
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/early_fusion/out_model_early_fusion_19_36.pth'#'out_KAIST_model/early_fusion_gnll_0305/out_model_early_fusion_best_gnll.pth'#'/home/jamie/Desktop/kaist_final_models/early_fusion_model_23/model_0039999.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'
elif data_gen == 'middle_fusion':
    cfg.INPUT.FORMAT = 'BGRTTT'
    cfg.INPUT.NUM_IN_CHANNELS = 6
    cfg.MODEL.PIXEL_MEAN += [135.438, 135.438, 135.438]  # normalization!!
    cfg.MODEL.PIXEL_STD += [1.00, 1.00, 1.00]
    #cfg.MODEL.WEIGHTS = '/home/jamie/Desktop/kaist_final_models/output_middle_fusion/model_0029999.pth'#'good_model/KAIST/RGB/out_model_iter_2500.pth'
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/middle_fusion/out_model_middle_fusion_gnll_14_48.pth'
# Init predictor
cfg.DATASETS.TEST = (dataset_test, )
cfg.MODEL.ROI_HEADS.ENABLE_GAUSSIANNLLOSS = True
cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LOGITS = True
predictor = DefaultPredictor(cfg)
from evalKAIST.evaluation_script import evaluate


# Read KIAST file list
img_folder = '../../../Datasets/KAIST/'
file_path = img_folder + 'KAIST_evaluation/data/kaist-rgbt/splits/test-all-20.txt'
with open(file_path) as f:contents = f.readlines()

img_folder = '../../../Datasets/KAIST/test/'
var_dict = {}
var_dict_name = out_folder + 'KAIST_' + data_gen + '_variance.npz'
with open(out_file_name, mode='w') as f:
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
            inputs = cv2.imread(path_rgb)
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

        #print('file = ',fpath)
    
        # Make prediction
        outputs = predictor(inputs)
        variance = outputs['instances']._fields['vars']
        var_dict[i+1] = variance

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
np.savez(var_dict_name, vars=var_dict)
evaluate('demo/evalKAIST/KAIST_annotation.json', out_file_name, 'Multispectral')