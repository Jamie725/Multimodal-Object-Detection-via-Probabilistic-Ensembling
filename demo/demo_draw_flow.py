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

# get path
#mypath = 'input/FLIR/Day/'
dataset = 'FLIR'
input_type = 'UVM'
train_or_val = 'val'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/RGB/'

files_names = [f for f in listdir(path) if isfile(join(path, f))]
out_folder = 'out/img/flow_UVM_scale_vis/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

torch.cuda.set_device(0)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Draw RGB
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
#cfg.MODEL.WEIGHTS = "good_model/mid_fusion/out_model_iter_42000.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# -------- Setting for 6 inputs -------- #
cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3 #4
#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
if input_type == 'UVV':
    cfg.MODEL.PIXEL_MEAN = [0.28809, 0.47052, 0.47052] #UVV
else:
    cfg.MODEL.PIXEL_MEAN = [11.2318, 7.2777, 14.8328] #UVM
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
#cfg.MODEL.WEIGHTS = "good_model/thermal_only/model_0009999.pth"
# -------------------------------------- #
#Draw trained thermal
if input_type == 'UVV':
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_scale_UVV_0414/out_model_iter_11000.pth'
else:
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_UVM_scale_0412/out_model_iter_22000.pth'
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

folder = '../../../Datasets/KAIST/test/KAIST_flow_test_sanitized/'
file_list = glob.glob(os.path.join(folder, '*.flo'))
img_folder = '../../../Datasets/KAIST/test/'
# Create predictor
predictor = DefaultPredictor(cfg)

scale = 0
for i in range(len(file_list)):
    fpath = file_list[i]
    flow = readFlow(fpath)
    
    image = np.zeros((flow.shape[0], flow.shape[1], 3))
    image[:,:,0] = flow[:,:,0]
    image[:,:,1] = flow[:,:,1]
    # UVV
    if input_type == 'UVV':     
        image[:,:,2] = flow[:,:,1]
        if scale == 1:
            image *= 3.0
            image += 128.0
            image[image>255] = 255.0
        else:            
            image = np.abs(image) / 40.0 * 255.0
            image[image>255] = 255.0
    else:        
        # UVM
        flow_s = flow * flow
        magnitude = np.sqrt(flow_s[:,:,0] + flow_s[:,:,1])
        image[:,:,2] = magnitude
        if scale == 1:
            image *= 3.0
            image += 128.0
            image[image>255] = 255.0
        else:
            image = np.abs(image) / 40.0 * 255.0
            image[image>255] = 255.0

    print('file = ',fpath)
    #pdb.set_trace()

    set_name = fpath.split('/')[-1].split('_')[0]
    V_name = fpath.split('/')[-1].split('_')[1]
    img_name = fpath.split('/')[-1].split('_')[2].split('.')[0] + '.jpg'
    img_path = img_folder + set_name + '/' + V_name + '/lwir/' + img_name
    img = cv2.imread(img_path)
    # Make prediction
    outputs = predictor(image)
    
    #print('name = ', files_names[i])
    out_name = out_folder +'/' + set_name + '_' + V_name + '_' + img_name
    #out_name = 'FLIR_08743_thermal.jpg'
    print(out_name)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('img_t',v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    v.save(out_name)
    #pdb.set_trace()