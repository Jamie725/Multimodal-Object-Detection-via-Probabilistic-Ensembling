# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer_paper import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.flow_utils import readFlow
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import torch
import glob

# ----- get path -----
dataset = 'FLIR'
input_type = 'thermal_only'
train_or_val = 'val'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/RGB/'
files_names = [f for f in listdir(path) if isfile(join(path, f))]

# Set CUDA
torch.cuda.set_device(0)

# Parameter settings
cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# -------- Setting for 6 inputs -------- #
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
if input_type == 'UVV':
    cfg.MODEL.PIXEL_MEAN = [0.28809, 0.47052, 0.47052] #UVV
elif input_type == 'UVM':
    cfg.MODEL.PIXEL_MEAN = [11.2318, 7.2777, 14.8328] #UVM
elif input_type == 'BGR':
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675] # BGR
# -------------------------------------- #
#Draw trained thermal
if input_type == 'UVV':
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_scale_UVV_0414/out_model_iter_11000.pth'
elif input_type == 'UVM':
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_UVM_scale_0412/out_model_iter_22000.pth'
elif input_type == 'BGR':
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.MODEL.WEIGHTS = 'good_model/KAIST/RGB/out_model_iter_2500.pth'#'out_training/KAIST_RGB/out_model_1000.pth'#'../../../kaist_final_models/output_RGB_59/model_0009999.pth'
elif input_type == 'thermal_only':
    #cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    #cfg.MODEL.WEIGHTS = 'out_training/KAIST_thermal_only/out_model_1000.pth'#'../../../kaist_final_models/output_RGB_59/model_0009999.pth'
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_thermal_only_1013/out_model_iter_6000.pth'
# Create predictor  
predictor = DefaultPredictor(cfg)

# Read KIAST file list
img_folder = '../../../Datasets/KAIST/'
file_path = img_folder + 'KAIST_evaluation/data/kaist-rgbt/splits/test-all-20.txt'
with open(file_path) as f:
    contents = f.readlines()

#folder = '../../../Datasets/KAIST/test/KAIST_flow_test_sanitized/'
#file_list = glob.glob(os.path.join(folder, '*.flo'))
img_folder = '../../../Datasets/KAIST/test/'
out_folder = 'out/box_predictions/KAIST/'
out_file_name = out_folder+'KAIST_'+input_type+'_det_1013_model.txt'

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

with open(out_file_name, mode='w') as f:
    for i in range(len(contents)):
        fpath = contents[i].split('\n')[0]            
        set_num = fpath.split('/')[0]
        V_num = fpath.split('/')[1]
        img_num = fpath.split('/')[2]
        img_path = img_folder + set_num + '/' + V_num + '/visible/' + img_num + '.jpg'
        img = cv2.imread(img_path)
        print('file = ',fpath)
    
        # Make prediction
        outputs = predictor(img)
        
        num_box = len(outputs['instances']._fields['pred_boxes'])
        #pdb.set_trace()
        #if num_box > 0:
        #    pdb.set_trace()
        for j in range(num_box):
            pred_class = outputs['instances'].pred_classes[j].cpu().numpy()
            if pred_class == 0:
                score = outputs['instances'].scores[j].cpu().numpy()*100                
                bbox = outputs['instances']._fields['pred_boxes'][j].tensor.cpu().numpy()[0]
                bbox[2] -= bbox[0] 
                bbox[3] -= bbox[1]
                
                f.write(str(i+1)+',')
                f.write(','.join(str(c) for c in bbox))
                #pdb.set_trace()
                f.write(','+str(score))
                f.write('\n')