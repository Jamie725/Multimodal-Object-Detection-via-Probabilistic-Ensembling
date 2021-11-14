# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer_paper import Visualizer
from detectron2.data import MetadataCatalog
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import torch

# get path
#mypath = 'input/FLIR/Day/'
dataset = 'FLIR'
train_or_val = 'val'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/resized_RGB/'

ratio = 1.4
files_names = [f for f in listdir(path) if isfile(join(path, f))]
out_folder = 'out/img/FLIR_mid_fusion_perturb_RGB/ratio_'+str(ratio)#bottom_right_ratio_'+str(ratio)
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

torch.cuda.set_device(1)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Draw RGB
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
#cfg.MODEL.WEIGHTS = "good_model/mid_fusion/out_model_iter_42000.pth"
# -------- Setting for 6 inputs -------- #
cfg.INPUT.FORMAT = 'BGRTTT'
cfg.INPUT.NUM_IN_CHANNELS = 6 #4
#cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#cfg.MODEL.WEIGHTS = "good_model/thermal_only/model_0009999.pth"
# -------------------------------------- #
#Draw trained thermal
#cfg.MODEL.WEIGHTS = os.path.join('output_val/good_model', "model_0009999.pth")
cfg.MODEL.WEIGHTS = os.path.join('good_model/3_class/mid_fusion/', "out_model_iter_100.pth")
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

# Create predictor
predictor = DefaultPredictor(cfg)

#for i in range(len(files_names)):
for i in range(1):
    # get image
    files_names[i] = 'FLIR_09012.jpg'
    #file_names[i] = 'FLIR_' + str(i) + '.jpg'
    path_t = '../../../Datasets/'+ dataset +'/'+train_or_val+'/thermal_8_bit/'    
    file_img = path_t + files_names[i].split(".")[0] + '.jpeg'
    img_t = cv2.imread(file_img)    

    file_RGB = path + files_names[i]
    img_rgb = cv2.imread(file_RGB)    
    width = int(640*ratio+0.5)
    height = int(512*ratio+0.5)
    img_rgb = cv2.resize(img_rgb, (width,height))
    img_rgb = img_rgb[:512, :640,:]
    
    out_file = out_folder + '/' + files_names[i].split('.')[0] + '_rgb_perturb.jpg'
    cv2.imwrite(out_file, img_rgb)
    #img_rgb = cv2.imread(file_RGB)
    
    img = np.zeros((512,640,6))
    img[:,:,:3] = img_rgb
    img[:,:,3:] = img_t[:,:,:]
    
    # Make prediction
    outputs = predictor(img)
    
    name = files_names[i].split('.')[0] + '.jpg'
    #print('name = ', files_names[i])
    out_name = out_folder +'/'+ name
    #out_name = 'FLIR_08743_thermal.jpg'
    print(out_name)

    v = Visualizer(img_t[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('img_t',v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    v.save(out_name)