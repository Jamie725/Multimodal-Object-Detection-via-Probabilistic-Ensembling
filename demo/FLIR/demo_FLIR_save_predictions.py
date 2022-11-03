# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import json
import torch

# get path
dataset = 'FLIR'
# Train path
train_path = '../../../Datasets/'+ dataset +'/train/'
train_folder = '../../../Datasets/FLIR/train/'
train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4_channel_no_dogs_3_class.json'

# Validation path
val_path = '../../../Datasets/'+ dataset +'/val/'
val_folder = '../../../Datasets/FLIR/val/'
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
# get path
#mypath = 'input/FLIR/Day/'
dataset_name = 'FLIR'
data_set = 'val'
RGB_path = '../../../Datasets/'+ dataset_name +'/'+data_set+'/RGB/'
t_path = '../../../Datasets/'+ dataset_name +'/'+data_set+'/thermal_8_bit/'
data_gen = 'thermal_only'#'early_fusion'#'thermal_only'#'mid_fusion'
print('==========================')
print('model:', data_gen)
print('==========================')
# Build image id dictionary
#val_file_name = 'thermal_annotations_4_channel_no_dogs_3_class.json'#'thermal_annotations_4_channel_no_dogs_3_class.json'#'thermal_annotations_4_channel_no_dogs_Night.json'#'RGB_annotations_4_channel_no_dogs.json'
#val_json_path = '../../../Datasets/FLIR/'+data_set+'/' + val_file_name
val_file_name = 'thermal_RGBT_pairs_3_class.json'
val_json_path = '../../../Datasets/FLIR/'+data_set+'/' + val_file_name
data = json.load(open(val_json_path, 'r'))
name_to_id_dict = {}
for i in range(len(data['images'])):
    file_name = data['images'][i]['file_name'].split('/')[1].split('.')[0]
    name_to_id_dict[file_name] = data['images'][i]['id']

# File names
files_names = [f for f in listdir(RGB_path) if isfile(join(RGB_path, f))]
out_folder = 'out/box_predictions/3_class/'

# Make folder if not exists
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LOGITS = True
cfg.MODEL.ROI_BOX_HEAD.DROP_OUT = True
cfg.MODEL.BACKBONE.FREEZE_AT = 3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
torch.cuda.set_device(0)

if data_gen == 'RGB':
    cfg.MODEL.WEIGHTS = "trained_models/Detectron2_pretrained/model_final_f6e8b1.pkl"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
elif data_gen == 'thermal_only':
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth'    
elif data_gen == 'early_fusion':
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/early_fusion/out_model_early_fusion.pth'
    cfg.INPUT.FORMAT = 'BGRT'
    cfg.INPUT.NUM_IN_CHANNELS = 4
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
elif data_gen == 'mid_fusion':
    cfg.MODEL.WEIGHTS = 'trained_models/FLIR/models/mid_fusion/out_model_middle_fusion.pth'
    cfg.INPUT.FORMAT = 'BGRTTT'
    cfg.INPUT.NUM_IN_CHANNELS = 6
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

torch.cuda.set_device(0)

# Create predictor
predictor = DefaultPredictor(cfg)
out_pred_file = out_folder+data_set+'_'+data_gen+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90_Night.json'
print('out file:', out_pred_file)
out_dicts = {}
image_dict = []
boxes_dict = []
scores_dict = []
classes_dict = []
class_logits_dict = []
prob_dict = []
img_id_dict = []
std_dict = []

for i in range(len(data['images'])):
    file_name = data['images'][i]['file_name'].split('/')[1].split('.')[0]
    RGB_file = RGB_path + file_name + '.jpg'
    thermal_file = t_path + file_name+'.jpeg'
    print('id: ', i)
    input_file = None
    if data_gen == 'RGB':
        input_file = RGB_file
        img = cv2.imread(input_file)
    elif data_gen == 'thermal_only':
        input_file = thermal_file
        img = cv2.imread(input_file)
    elif data_gen == 'early_fusion':
        input_file_thermal = thermal_file
        thermal_img = cv2.imread(input_file_thermal)
        input_file_RGB = RGB_file
        rgb_img = cv2.imread(input_file_RGB)
        rgb_img = cv2.resize(rgb_img, (thermal_img.shape[1],thermal_img.shape[0]), cv2.INTER_CUBIC)
        img = np.zeros((thermal_img.shape[0], thermal_img.shape[1], 4))
        img [:,:,0:3] = rgb_img
        img [:,:,-1] = thermal_img[:,:,0]
    elif data_gen == 'mid_fusion':
        input_file_thermal = thermal_file
        thermal_img = cv2.imread(input_file_thermal)
        input_file_RGB = RGB_file
        rgb_img = cv2.imread(input_file_RGB)
        rgb_img = cv2.resize(rgb_img, (thermal_img.shape[1],thermal_img.shape[0]), cv2.INTER_CUBIC)        
        img = np.zeros((thermal_img.shape[0], thermal_img.shape[1], 6))
        img [:,:,0:3] = rgb_img
        img [:,:,3:6] = thermal_img
        
    predictor.model.eval()
    for m in predictor.model.modules():    
        if m.__class__.__name__.startswith('Dropout'):            
            m.train()    

    iters = 20
    
    # Make prediction
    predictor.model.eval()
    try:
        predict = predictor(img)
    except:
        pdb.set_trace()
    
    predictions = predict['instances'].to('cpu')
    boxes = predictions.pred_boxes.tensor.tolist() if predictions.has("pred_boxes") else None
    scores = predictions.scores.tolist() if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    class_logits = predictions.class_logits.tolist() if predictions.has("class_logits") else None
    probs = predictions.prob_score.tolist() if predictions.has("prob_score") else None
    #stds = predictions.stds.tolist() if predictions.has("stds") else None
    
    out_boxes = []
    out_scores = []
    out_classes = []
    out_logits = []
    out_probs = []
    #out_stds = []

    for j in range(len(boxes)):
        if classes[j] <= 2:
            out_boxes.append(boxes[j])
            out_scores.append(scores[j])
            out_classes.append(classes[j])
            out_logits.append(class_logits[j])
            out_probs.append(probs[j])
            #out_stds.append(stds[j])

    #out_boxes = np.array(out_boxes)
    #out_scores = np.array(out_scores)
    #out_classes = np.array(out_classes)
    image_dict.append(files_names[i])
    boxes_dict.append(out_boxes)
    scores_dict.append(out_scores)
    classes_dict.append(out_classes)
    class_logits_dict.append(out_logits)
    prob_dict.append(out_probs)
    #std_dict.append(out_stds)
    pdb.set_trace()
    try:
        img_id_dict.append(name_to_id_dict[file_name])
    except:
        pdb.set_trace()

out_dicts['image'] = image_dict
out_dicts['boxes'] = boxes_dict
out_dicts['scores'] = scores_dict
out_dicts['classes'] = classes_dict
out_dicts['image_id'] = img_id_dict
out_dicts['class_logits'] = class_logits_dict
out_dicts['probs'] = prob_dict
#out_dicts['stds'] = std_dict

with open(out_pred_file, 'w') as outfile:
    json.dump(out_dicts, outfile, indent=2)
    print('Finish predictions!')