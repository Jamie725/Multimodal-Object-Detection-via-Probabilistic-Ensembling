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
input_type = 'BGRTUV'
train_or_val = 'val'
path = '../../../Datasets/'+ dataset +'/'+train_or_val+'/RGB/'
files_names = [f for f in listdir(path) if isfile(join(path, f))]
out_folder = 'out/img/flow_UVM_scale_vis/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# Set CUDA
torch.cuda.set_device(1)

# Parameter settings
cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.INPUT.FORMAT = 'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 3
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
if input_type == 'UVV':
    cfg.MODEL.PIXEL_MEAN = [0.28809, 0.47052, 0.47052] #UVV
else:
    cfg.MODEL.PIXEL_MEAN = [11.2318, 7.2777, 14.8328] #UVM
# -------------------------------------- #
# -------- Setting for 6 inputs -------- #
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.INPUT.FORMAT = 'BGRTUV'#'BGR'
cfg.INPUT.NUM_IN_CHANNELS = 6
cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 11.2318+128, 7.2777+128]#[225.328, 226.723, 235.070]#[103.530, 116.280, 123.675]
cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#cfg.MODEL.BLUR_RGB = True
cfg.MODEL.MAX_POOL_RGB = False

#Draw trained thermal
if input_type == 'UVV':
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_scale_UVV_0414/out_model_iter_11000.pth'
elif input_type == 'UVM':
    #cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_UVM_scale_0412/out_model_iter_22000.pth'
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_UVM_scale_0421/out_model_iter_32000.pth'
elif input_type == 'BGRTUV':
    cfg.MODEL.WEIGHTS = 'out_training/KAIST_flow_BGRTUV_0428/out_model_iter_21000.pth'
# Create predictor
predictor = DefaultPredictor(cfg)

# Read KIAST file list
img_folder = '../../../Datasets/KAIST/'
file_path = img_folder + 'KAIST_evaluation/data/kaist-rgbt/splits/test-all-20.txt'

with open(file_path) as f:
    contents = f.readlines()

folder = '../../../Datasets/KAIST/test/KAIST_flow_test_sanitized/'
file_list = glob.glob(os.path.join(folder, '*.flo'))
img_folder = '../../../Datasets/KAIST/test/'

out_folder = 'out/box_predictions/KAIST/'
out_file_name = out_folder+'KAIST_flow_'+input_type+'_BGRTUV_det.txt'

with open(out_file_name, mode='w') as f:
    """
    for i in range(len(all_boxes[1])):
      for box in all_boxes[1][i]:          
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        if box[3] < ignore_thresh:
          continue
        # rescale score 0~100
        box[4] *= 100
        f.write(str(i+1)+',')
        f.write(','.join(str(c) for c in box))
        f.write('\n')
    """

    scale = 1
    for i in range(len(contents)):
        """
        fpath = contents[i].split('\n')[0]
        set_num = fpath.split('/')[0]
        V_num = fpath.split('/')[1]
        img_num = fpath.split('/')[2]
        flow_path = folder + set_num + '_' + V_num + '_' + img_num + '.flo'
        flow = readFlow(flow_path)
        
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
        """
        
        flow_folder = '../../../Datasets/KAIST/test/KAIST_flow_test_sanitized/'
        img_folder = '../../../Datasets/KAIST/test/'
        
        fpath = contents[i].split('\n')[0]
        set_num = fpath.split('/')[0]
        V_num = fpath.split('/')[1]
        img_num = fpath.split('/')[2]
        flow_path = folder + set_num + '_' + V_num + '_' + img_num + '.flo'
        flow = readFlow(flow_path)
        
        image = np.zeros((flow.shape[0], flow.shape[1], 6))
        image[:,:,4] = flow[:,:,0]
        image[:,:,5] = flow[:,:,1]    
        image *= 3
        image += 128.0
        image[image>255] = 255.0
        
        set_name = fpath.split('/')[0]
        V_name = fpath.split('/')[1]
        img_name = fpath.split('/')[2]

        fname_bgr = img_folder + set_name + '/' + V_name + '/visible/' + img_name + '.jpg'
        fname_thr = img_folder + set_name + '/' + V_name + '/lwir/' + img_name + '.jpg'
            
        bgr = cv2.imread(fname_bgr)
        thr = cv2.imread(fname_thr)

        image[:,:,0:3] = bgr
        image[:,:,3] = thr[:,:,0]

        print('file = ',fpath)
        
        """
        set_name = fpath.split('/')[-1].split('_')[0]
        V_name = fpath.split('/')[-1].split('_')[1]
        img_name = fpath.split('/')[-1].split('_')[2].split('.')[0] + '.jpg'
        img_path = img_folder + set_name + '/' + V_name + '/lwir/' + img_name
        img = cv2.imread(img_path)
        """
        # Make prediction
        outputs = predictor(image)

        num_box = len(outputs['instances']._fields['pred_boxes'])

        for j in range(num_box):
            score = outputs['instances'].scores[j].cpu().numpy()*100
            pred_class = outputs['instances'].pred_classes[j]
            bbox = outputs['instances']._fields['pred_boxes'][j].tensor.cpu().numpy()[0]
            bbox[2] -= bbox[0] 
            bbox[3] -= bbox[1] 
            #pdb.set_trace()
            
            f.write(str(i+1)+',')
            f.write(','.join(str(c) for c in bbox))
            #pdb.set_trace()
            f.write(','+str(score))
            f.write('\n')
        """
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
        """