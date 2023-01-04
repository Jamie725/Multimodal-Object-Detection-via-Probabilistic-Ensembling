# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.opt import config_parser
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import os
import json
import torch


def save_predictions(args):
    # Dataset information
    dataset_name = args.dataset_name
    val_folder = args.dataset_path
    val_json_path = args.dataset_path + '/FLIR_thermal_RGBT_pairs_val.json'
    # Test on validation set
    register_coco_instances(dataset_name, {}, val_json_path, val_folder)
    FLIR_metadata_test = MetadataCatalog.get(dataset_name)
    dataset_dicts_test = DatasetCatalog.get(dataset_name)
    
    RGB_path = val_folder + '/RGB/'
    t_path = val_folder + '/thermal_8_bit/'
    method = args.fusion_method
    print('==========================')
    print('model:', method)
    print('==========================')
    # Build image id dictionary
    
    data = json.load(open(val_json_path, 'r'))
    name_to_id_dict = {}
    for i in range(len(data['images'])):
        file_name = data['images'][i]['file_name'].split('/')[1].split('.')[0]
        name_to_id_dict[file_name] = data['images'][i]['id']

    # File names
    files_names = [f for f in listdir(RGB_path) if isfile(join(RGB_path, f))]
    out_folder = args.outfolder

    # Make folder if not exists
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LOGITS = True
    cfg.MODEL.ROI_BOX_HEAD.DROP_OUT = True
    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.ENABLE_GAUSSIANNLLOSS = True

    if method == 'rgb_only':
        cfg.MODEL.WEIGHTS = "trained_models/Detectron2_pretrained/model_final_f6e8b1.pkl"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    elif method == 'thermal_only':
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    elif method == 'early_fusion':
        cfg.INPUT.FORMAT = 'BGRT'
        cfg.INPUT.NUM_IN_CHANNELS = 4
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438]
        cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
    elif method == 'middle_fusion':
        cfg.INPUT.FORMAT = 'BGRTTT'
        cfg.INPUT.NUM_IN_CHANNELS = 6
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675, 135.438, 135.438, 135.438]
        cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        print("The method is not supported in this code!")

    print("model loaded:", cfg.MODEL.WEIGHTS)
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    out_pred_file = os.path.join(out_folder, 'val_'+method+'_predictions.json')
    print('out file:', out_pred_file)
    out_dicts = {}
    image_dict = []
    boxes_dict = []
    scores_dict = []
    classes_dict = []
    class_logits_dict = []
    prob_dict = []
    img_id_dict = []
    var_dict = []

    for i in tqdm(range(len(data['images']))):
        file_name = data['images'][i]['file_name'].split('/')[1].split('.')[0]
        RGB_file = RGB_path + file_name + '.jpg'
        thermal_file = t_path + file_name+'.jpeg'
        input_file = None
        if method == 'RGB':
            input_file = RGB_file
            img = cv2.imread(input_file)
        elif method == 'thermal_only':
            input_file = thermal_file
            img = cv2.imread(input_file)
        elif method == 'early_fusion':
            input_file_thermal = thermal_file
            thermal_img = cv2.imread(input_file_thermal)
            input_file_RGB = RGB_file
            rgb_img = cv2.imread(input_file_RGB)
            rgb_img = cv2.resize(rgb_img, (thermal_img.shape[1],thermal_img.shape[0]), cv2.INTER_CUBIC)
            img = np.zeros((thermal_img.shape[0], thermal_img.shape[1], 4))
            img [:,:,0:3] = rgb_img
            img [:,:,-1] = thermal_img[:,:,0]
        elif method == 'middle_fusion':
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
        
        # Make prediction
        predictor.model.eval()    
        predict = predictor(img)
        
        
        predictions = predict['instances'].to('cpu')
        boxes = predictions.pred_boxes.tensor.tolist() if predictions.has("pred_boxes") else None
        scores = predictions.scores.tolist() if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        class_logits = predictions.class_logits.tolist() if predictions.has("class_logits") else None
        probs = predictions.prob_score.tolist() if predictions.has("prob_score") else None
        variances = predictions.vars.tolist() if predictions.has('vars') else None
        
        out_boxes = []
        out_scores = []
        out_classes = []
        out_logits = []
        out_probs = []
        out_vars = []

        for j in range(len(boxes)):
            if classes[j] <= 2:
                out_boxes.append(boxes[j])
                out_scores.append(scores[j])
                out_classes.append(classes[j])
                out_logits.append(class_logits[j])
                out_probs.append(probs[j])
                out_vars.append(variances[j])
                
        image_dict.append(files_names[i])
        boxes_dict.append(out_boxes)
        scores_dict.append(out_scores)
        classes_dict.append(out_classes)
        class_logits_dict.append(out_logits)
        prob_dict.append(out_probs)
        img_id_dict.append(name_to_id_dict[file_name])
        var_dict.append(out_vars)
        
    out_dicts['image'] = image_dict
    out_dicts['boxes'] = boxes_dict
    out_dicts['scores'] = scores_dict
    out_dicts['classes'] = classes_dict
    out_dicts['image_id'] = img_id_dict
    out_dicts['class_logits'] = class_logits_dict
    out_dicts['probs'] = prob_dict
    out_dicts['vars'] = var_dict

    with open(out_pred_file, 'w') as outfile:
        json.dump(out_dicts, outfile, indent=2)


if __name__ == '__main__':
    args = config_parser()
    save_predictions(args)