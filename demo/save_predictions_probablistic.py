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

def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels

def nms_calc_uncertainty(final_dets, final_scores, dets, scores, thresh):
    import numpy as np
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = []
    for i in range(len(final_scores)):
        id = np.where((final_scores[i] - scores) == 0)[0][0]
        order.append(id)
        
    order = np.array(order)
    #order = scores.argsort()[::-1]
    
    keep = []
    stds = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[:])
        yy1 = np.maximum(y1[i], y1[:])
        xx2 = np.minimum(x2[i], x2[:])
        yy2 = np.minimum(y2[i], y2[:])
        
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        inter = width * height        
        ovr = inter / (areas[i] + areas[:] - inter)
        
        # Find bbox with overlapping over threshold
        inds = np.where(ovr >= thresh)[0]
        ovrlp_boxes = dets[inds]
        std = np.std(ovrlp_boxes, axis=0)
        avg_std = np.mean(std)
        stds.append(avg_std)        
        order = order[1:]    
    return stds

def calc_sigma(predictor, img, iters):
    box_all = np.zeros((1,4))
    scores_all = np.zeros((1,1))
    for i in range(iters):
        pred = predictor(img)
        predictions = pred['instances'].to('cpu')
        boxes = predictions.pred_boxes.tensor.tolist() if predictions.has("pred_boxes") else None
        scores = predictions.pred_boxes.tensor.tolist() if predictions.has("scores") else None
        boxes = np.array(boxes)
        box_all = np.vstack((box_all, boxes))
    predictor.model.eval()
    final_dets = predictor(img)
    ##nms_calc_uncertainty(final_dets, final_scores, dets, scores, thresh)
    pdb.set_trace()
    return None

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
val_file_name = 'thermal_RGBT_pairs_3_class_Night.json'
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
cfg.MODEL.ROI_HEADS.ESTIMATE_UNCERTAINTY = True

"""
# Train config
cfg.DATASETS.TRAIN = (dataset_train,)
cfg.DATASETS.TEST = (dataset_test, )
#cfg.TEST.EVAL_PERIOD = 50
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

###### Performance tuning ########
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000
# Set GPU: pid, 1583
"""
torch.cuda.set_device(0)

cfg.MODEL.ROI_BOX_HEAD.DROP_OUT = True
cfg.MODEL.BACKBONE.FREEZE_AT = 3

if data_gen == 'RGB':
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
elif data_gen == 'thermal_only':
    #cfg.MODEL.WEIGHTS = 'output_val/good_model/model_0009999.pth'
    #cfg.MODEL.WEIGHTS = 'good_model/3_class/thermal_only/out_model_iter_15000.pth'
    cfg.MODEL.WEIGHTS = 'good_model/3_class/thermal_only/out_model_thermal_only_dropout.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
elif data_gen == 'early_fusion':
    #cfg.MODEL.WEIGHTS = 'good_model/early_fusion/out_model_iter_12000.pth'
    #cfg.MODEL.WEIGHTS = 'good_model/3_class/early_fusion/out_model_iter_100.pth'
    cfg.MODEL.WEIGHTS = 'good_model/3_class/early_fusion/out_model_early_fusion_dropout.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.INPUT.FORMAT = 'BGRT'
    cfg.INPUT.NUM_IN_CHANNELS = 4
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
elif data_gen == 'mid_fusion':
    #cfg.MODEL.WEIGHTS = 'good_model/mid_fusion/out_model_iter_42000.pth'
    #cfg.MODEL.WEIGHTS = 'good_model/3_class/mid_fusion/out_model_iter_100.pth'
    cfg.MODEL.WEIGHTS = 'good_model/3_class/mid_fusion/output_middle_fusion_drouout.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.INPUT.FORMAT = 'BGRTTT'
    cfg.INPUT.NUM_IN_CHANNELS = 6 #4
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
elif data_gen == 'BGR_only':
    cfg.MODEL.WEIGHTS = "out_BGR_only_1004/out_model_iter_46000.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]


torch.cuda.set_device(0)

# Create predictor
predictor = DefaultPredictor(cfg)
# Create trainer
#trainer = DefaultTrainer(cfg)

valid_class = [0, 1, 2]
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
#for i in range(8863,10228):
    # get image
    file_name = data['images'][i]['file_name'].split('/')[1].split('.')[0]
    RGB_file = RGB_path + file_name + '.jpg'
    thermal_file = t_path + file_name+'.jpeg'
    #RGB_file = RGB_path + 'FLIR_09062.jpg'
    #thermal_file = t_path + files_names[i].split('.')[0]+'.jpeg'
    print('id: ', i)
    input_file = None
    if data_gen == 'RGB' or data_gen == 'BGR_only':
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
        try:
            rgb_img = cv2.resize(rgb_img, (thermal_img.shape[1],thermal_img.shape[0]), cv2.INTER_CUBIC)
        except:
            pdb.set_trace()
        img = np.zeros((thermal_img.shape[0], thermal_img.shape[1], 6))
        img [:,:,0:3] = rgb_img
        img [:,:,3:6] = thermal_img
        
    #trainer.model.eval()
    predictor.model.eval()
    for m in predictor.model.modules():    
        if m.__class__.__name__.startswith('Dropout'):            
            m.train()    

    iters = 20
    stds = calc_sigma(predictor, img, iters)
    
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
    stds = predictions.stds.tolist() if predictions.has("stds") else None
    
    #labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
    #keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    out_boxes = []
    out_scores = []
    out_classes = []
    out_logits = []
    out_probs = []
    out_stds = []

    for j in range(len(boxes)):
        if classes[j] <= 2:
            out_boxes.append(boxes[j])
            out_scores.append(scores[j])
            out_classes.append(classes[j])
            out_logits.append(class_logits[j])
            out_probs.append(probs[j])
            out_stds.append(stds[j])

    #out_boxes = np.array(out_boxes)
    #out_scores = np.array(out_scores)
    #out_classes = np.array(out_classes)
    image_dict.append(files_names[i])
    boxes_dict.append(out_boxes)
    scores_dict.append(out_scores)
    classes_dict.append(out_classes)
    class_logits_dict.append(out_logits)
    prob_dict.append(out_probs)
    std_dict.append(out_stds)

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
out_dicts['stds'] = std_dict

with open(out_pred_file, 'w') as outfile:
    json.dump(out_dicts, outfile, indent=2)
    print('Finish predictions!')