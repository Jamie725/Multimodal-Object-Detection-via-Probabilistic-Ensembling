import pdb
import os
import json
import numpy as np
from os.path import isfile, join
import cv2
import torch
import pickle
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances, Boxes
from detectron2.evaluation import FLIREvaluator
# For COCO evaluation
from fvcore.common.file_io import PathManager
from detectron2.pycocotools.coco import COCO
from detectron2.pycocotools.cocoeval import COCOeval

def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    #pdb.set_trace()
    return keep

def visualize_2_frames(path_1, rgb_img_name, path_2, t_img_name, closest_id, rgb_box, t_box, out_name):
    rgb_img = cv2.imread(path_1 + rgb_img_name)
    t_img = cv2.imread(path_2 + t_img_name)
    
    resize_rgb = cv2.resize(rgb_img, (640, 512))
    resize_t = cv2.resize(t_img, (640, 512))
    out_img = np.zeros((512, 640*2, 3))

    t_box_match = t_box[closest_id]

    #image = cv2.circle(image, center_coordinates, 2, (0,0,255), 2)
    rect_rgb = cv2.rectangle(resize_rgb,(int(rgb_box[0]+0.5),int(rgb_box[1]+0.5)),(int(rgb_box[2]+0.5),int(rgb_box[3]+0.5)),(0,0,255),2)
    rect_t = cv2.rectangle(resize_t,(int(t_box_match[0]+0.5),int(t_box_match[1]+0.5)),(int(t_box_match[2]+0.5),int(t_box_match[3]+0.5)),(0,255,0),2)

    out_img[:, :640, :] = rect_rgb
    out_img[:, 640:, :] = rect_t
    out_img = cv2.rectangle(out_img,(640+int(rgb_box[0]+0.5),int(rgb_box[1]+0.5)),(640+int(rgb_box[2]+0.5),int(rgb_box[3]+0.5)),(0,0,255),2)
    cv2.imwrite(out_name, out_img)

def get_box_area(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area


def avg_bbox_fusion(match_bbox_vec):
    avg_bboxs = np.sum(match_bbox_vec,axis=0) / len(match_bbox_vec)
    return avg_bboxs

def bayesian_fusion(match_score_vec):
    log_positive_scores = np.log(match_score_vec)
    log_negative_scores = np.log(1 - match_score_vec)
    fused_positive = np.exp(np.sum(log_positive_scores))
    fused_negative = np.exp(np.sum(log_negative_scores))
    fused_positive_normalized = fused_positive / (fused_positive + fused_negative)
    return fused_positive_normalized

def nms_1(info_1, info_2):
    # RGB boxes append thermal boxes
    # Order: len(RGB) | len(thermal)
    # Boxes
    boxes = info_1['bbox'].copy()
    boxes.extend(info_2['bbox'])
    boxes = torch.Tensor(boxes)
    # Scores
    scores = info_1['score'].copy()
    scores.extend(info_2['score'])
    scores = torch.Tensor(scores)
    # Classes
    classes = info_1['class'].copy()
    classes.extend(info_2['class'])
    classes = torch.Tensor(classes)
    # Perform nms
    iou_threshold = 0.5

    #keep_id = box_ops.batched_nms(boxes, scores, classes, iou_threshold)
    try:
        keep_id = batched_nms(boxes, scores, classes, iou_threshold)
    except:
        pdb.set_trace()
    # Add to output
    out_boxes = boxes[keep_id]
    out_scores = torch.Tensor(scores[keep_id])
    out_class = torch.Tensor(classes[keep_id])
    return out_boxes, out_scores, out_class

def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight[i] * bbox[i]
    return out_bbox

def prepare_data(info1, info2):
    bbox1 = np.array(info1['bbox'])
    bbox2 = np.array(info2['bbox'])
    score1 = np.array(info1['score'])
    score2 = np.array(info2['score'])
    class1 = np.array(info1['class'])
    class2 = np.array(info2['class'])
    
    out_bbox = np.concatenate((bbox1, bbox2), axis=0)
    out_score = np.concatenate((score1, score2), axis=0)
    out_class = np.concatenate((class1, class2), axis=0)

    return out_bbox, out_score, out_class

def nms_2(dets, score, classes, thresh, method):
    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    scores = score#dets[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    match_scores = []
    match_bboxs = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #print(dets[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= thresh)[0]
        match = np.where(ovr > thresh)[0]
        match_ind = order[match+1]
        match_score = list(score[match_ind])
        match_bbox = list(dets[match_ind][:,:4])
        original_score = score[i].tolist()
        original_bbox = dets[i][:4]
        if len(match_score)>0:
            match_score += [original_score]
            #pdb.set_trace()
            if method == 'avg_score':
                final_score = np.mean(np.asarray(match_score))
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'baysian':
                final_score = bayesian_fusion(np.asarray(match_score))
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'baysian_avg_bbox':
                final_score = bayesian_fusion(np.asarray(match_score))
                match_bbox += [original_bbox]
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'baysian_wt_score_box':
                final_score = bayesian_fusion(np.asarray(match_score))
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_score)

            #final_bbox = avg_bbox_fusion(match_bbox)
            match_scores.append(final_score)
            match_bboxs.append(final_bbox)
        else:
            match_scores.append(original_score)
            match_bboxs.append(original_bbox)
        
        #print(match_scores)
        order = order[inds + 1]
        
    #pdb.set_trace()
    assert len(keep)==len(match_scores)
    assert len(keep)==len(match_bboxs)

    match_bboxs = match_bboxs
    match_scores = torch.Tensor(match_scores)
    match_classes = torch.Tensor(classes[keep])
    return keep,match_scores,match_bboxs, match_classes

def draw_box(img, bbox, color):
    for i in range(len(bbox)):
        img = cv2.rectangle(img,  (int(bbox[i][0]+0.5), int(bbox[i][1]+0.5)),  (int(bbox[i][2]+0.5), int(bbox[i][3]+0.5)), color, 2)
    return img
def apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, method):
    evaluator.reset()
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    
    num_img = len(det_2['image'])
    count_1=0
    count_2=0
    count_fusion=0

    print('Method: ', method)

    for i in range(num_img):
        info_1 = {}
        info_1['img_name'] = det_1['image'][i]
        info_1['bbox'] = det_1['boxes'][i]
        info_1['score'] = det_1['scores'][i]
        info_1['class'] = det_1['classes'][i]
        
        info_2 = {}
        info_2['img_name'] = det_2['image'][i].split('.')[0] + '.jpeg'
        info_2['bbox'] = det_2['boxes'][i]
        info_2['score'] = det_2['scores'][i]
        info_2['class'] = det_2['classes'][i]
        #pdb.set_trace()
        if len(info_1['bbox']) == 0 or len(info_2['bbox']) == 0:
            if(len(info_1['bbox']) > 0):
                out_boxes = np.array(info_1['bbox'])
                out_class = torch.Tensor(info_1['class'])
                out_scores = torch.Tensor(info_1['score'])
            elif(len(info_2['bbox']) > 0):
                out_boxes = np.array(info_2['bbox'])
                out_class = torch.Tensor(info_2['class'])
                out_scores = torch.Tensor(info_2['score'])
            else:
                out_boxes = np.array(info_2['bbox'])
                out_class = torch.Tensor(info_2['class'])
                out_scores = torch.Tensor(info_2['score'])
        else:
            if method == 'nms':            
                out_boxes, out_scores, out_class = nms_1(info_1, info_2)
            elif method == 'pooling':
                in_boxes, in_scores, in_class = prepare_data(info_1, info_2)
                out_boxes = in_boxes
                out_scores = torch.Tensor(in_scores)
                out_class = torch.Tensor(in_class)
            elif method == 'baysian' or method == 'baysian_avg_bbox' or method == 'avg_score' or method == 'baysian_wt_score_box':
                threshold = 0.5
                in_boxes, in_scores, in_class = prepare_data(info_1, info_2)
                keep, out_scores, out_boxes, out_class = nms_2(in_boxes, in_scores, in_class, threshold, method)
        
        count_1 += len(info_1['bbox'])
        count_2 += len(info_2['bbox'])
        count_fusion += len(out_boxes)
       
        
        file_name = img_folder + info_1['img_name'].split('.')[0] + '.jpeg'
        img = cv2.imread(file_name)
        H, W, _ = img.shape

        # Handle inputs
        inputs = []
        input_info = {}
        input_info['file_name'] = file_name
        input_info['height'] = H
        input_info['width'] = W
        input_info['image_id'] = det_2['image_id'][i]
        input_info['image'] = torch.Tensor(img)
        inputs.append(input_info)

        # Handle outputs
        outputs = []
        out_info = {}
        proposals = Instances([H, W])
        proposals.pred_boxes = Boxes(out_boxes)
        proposals.scores = out_scores
        proposals.pred_classes = out_class
        out_info['instances'] = proposals
        outputs.append(out_info)
        evaluator.process(inputs, outputs)

        img = draw_box(img, out_boxes, (0,255,0))
        out_img_name = 'out_img_baysian_fusion/' + file_name.split('thermal_8_bit/')[1].split('.')[0]+'_baysian_avg_bbox.jpg'
        #cv2.imwrite(out_img_name, img)
        #pdb.set_trace()
        """
        if '09115' in file_name:
            out_img_name = 'out_img_baysian_fusion/' + file_name.split('thermal_8_bit/')[1].split('.')[0]+'_baysian_avg_bbox.jpg'
            pdb.set_trace()
            cv2.imwrite(out_img_name, img)
        """
        
    results = evaluator.evaluate(out_eval_path='FLIR_pooling_.out')
    
    if results is None:
        results = {}
    
    avgRGB = count_1 / num_img
    avgThermal = count_2 / num_img
    avgNMS = count_fusion / num_img

    print('Avg bbox for RGB:', avgRGB, "average count thermal:", avgThermal, 'average count nms:', avgNMS)
    return results


if __name__ == '__main__':
    data_set = 'val'
    data_folder = 'out/box_predictions/'
    dataset = 'FLIR'
    IOU = 50
    time = 'all'
    logit = False
    if time == 'Day':
        val_file_name = 'thermal_annotations_4_channel_no_dogs_Day.json'#'RGB_annotations_4_channel_no_dogs.json'#'thermal_annotations_4_channel_no_dogs_Day.json'#
        if logit:
            det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_'+time+'_with_logits.json'
            det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_'+time+'_with_logits.json'
        else:
            det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_'+time+'.json'
            det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_'+time+'.json'
    elif time == 'Night':
        val_file_name = 'thermal_annotations_4_channel_no_dogs_Night.json'
        if logit:
            det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_'+time+'_with_logits.json'
            det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_'+time+'_with_logits.json'
        else:
            det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_'+time+'.json'
            det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_'+time+'.json'
    elif time == 'all':
        val_file_name = 'RGB_annotations_4_channel_no_dogs.json'
        if logit:
            det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_with_logits.json'
            det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_with_logits.json'
        else:
            det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50.json'
            det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50.json'
        
    #det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50.json'#'val_thermal_only_predictions_IOU50_day.json'#
    #det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50.json'
    path_1 = '../../../Datasets/FLIR/' + data_set + '/resized_RGB/'
    path_2 = '../../../Datasets/FLIR/' + data_set + '/thermal_8_bit/'
    out_folder = 'out/box_comparison/'
    #train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4_channel_no_dogs.json'
    #val_file_name = 'RGB_annotations_4_channel_no_dogs.json'#'RGB_annotations_4_channel_no_dogs.json'#'thermal_annotations_4_channel_no_dogs_day.json'#
    val_json_path = '../../../Datasets/'+dataset+'/val/' + val_file_name
    val_folder = '../../../Datasets/FLIR/val/thermal_8_bit'

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Register dataset
    dataset = 'FLIR_val'
    register_coco_instances(dataset, {}, val_json_path, val_folder)
    FLIR_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = DatasetCatalog.get(dataset)

    # Create config
    cfg = get_cfg()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = out_folder
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "good_model/out_model_iter_32000.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.DATASETS.TEST = (dataset, )
    cfg.INPUT.FORMAT = 'BGR'
    cfg.INPUT.NUM_IN_CHANNELS = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # Read detection results
    det_1 = json.load(open(det_file_1, 'r'))
    det_2 = json.load(open(det_file_2, 'r'))
    evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='out/mAP/FLIR_Baysian_'+time+'_1.out')
    #result = apply_late_fusion_and_evaluate(evaluator, det_1, det_2, 'nms')
    method = 'baysian_wt_score_box'#'baysian_wt_score_box'#'baysian_avg_bbox'#'avg_score'#'pooling' #'baysian'#'nms'
    result = apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, method)