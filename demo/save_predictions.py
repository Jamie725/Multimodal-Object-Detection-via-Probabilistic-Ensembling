# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import pdb
import json

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
# get path
#mypath = 'input/FLIR/Day/'
dataset_name = 'FLIR'
data_set = 'val'
RGB_path = '../../../Datasets/'+ dataset_name +'/'+data_set+'/resized_RGB/'
t_path = '../../../Datasets/'+ dataset_name +'/'+data_set+'/thermal_8_bit/'
data_gen = 'RGB'

# Build image id dictionary
val_json_path = '../../../Datasets/FLIR/val/RGB_annotations_4_channel_no_dogs.json'
data = json.load(open(val_json_path, 'r'))
name_to_id_dict = {}
for i in range(len(data['images'])):
    file_name = data['images'][i]['file_name'].split('/')[1].split('.')[0]
    name_to_id_dict[file_name] = data['images'][i]['id']

# File names
files_names = [f for f in listdir(RGB_path) if isfile(join(RGB_path, f))]
out_folder = 'out/mAP'
# Make folder if not exists
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

if data_gen == 'RGB':
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
else:
    cfg.MODEL.WEIGHTS = 'output_val/good_model/model_0009999.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

# Create predictor
predictor = DefaultPredictor(cfg)

valid_class = [0, 1, 2]
out_pred_file = 'out/mAP/'+data_set+'_'+data_gen+'_predictions_IOU50.json'

out_dicts = {}
image_dict = []
boxes_dict = []
scores_dict = []
classes_dict = []
img_id_dict = []
    
for i in range(len(files_names)):
    # get image
    RGB_file = RGB_path + files_names[i]
    thermal_file = t_path + files_names[i].split('.')[0]+'.jpeg'
    print('id: ', i)
    input_file = None
    if data_gen == 'RGB':
        input_file = RGB_file
    else:
        input_file = thermal_file
    
    img = cv2.imread(input_file)
    print('file = ', input_file)

    # Make prediction
    predict = predictor(img)
    predictions = predict['instances'].to('cpu')
    
    boxes = predictions.pred_boxes.tensor.tolist() if predictions.has("pred_boxes") else None
    scores = predictions.scores.tolist() if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    
    #labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
    #keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    out_boxes = []
    out_scores = []
    out_classes = []
    
    for j in range(len(boxes)):
        if classes[j] <= 2:
            out_boxes.append(boxes[j])
            out_scores.append(scores[j])
            out_classes.append(classes[j])

    #out_boxes = np.array(out_boxes)
    #out_scores = np.array(out_scores)
    #out_classes = np.array(out_classes)
    image_dict.append(files_names[i])
    boxes_dict.append(out_boxes)
    scores_dict.append(out_scores)
    classes_dict.append(out_classes)
    img_id_dict.append(name_to_id_dict[files_names[i].split('.')[0]])

out_dicts['image'] = image_dict
out_dicts['boxes'] = boxes_dict
out_dicts['scores'] = scores_dict
out_dicts['classes'] = classes_dict
out_dicts['image_id'] = img_id_dict

with open(out_pred_file, 'w') as outfile:
    json.dump(out_dicts, outfile, indent=2)
    print('Finish predictions!')