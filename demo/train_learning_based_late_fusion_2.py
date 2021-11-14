"""
Take 3 model as input

Correct multi-class Bayesian fusion

Fuse multi-class probability, also perform logits summation
"""
import pdb
import os
import json
import numpy as np
from os.path import isfile, join
import cv2
import torch
import pickle
from torch.nn import functional as F
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

def get_box_area(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area

def avg_bbox_fusion(match_bbox_vec):
    avg_bboxs = np.sum(match_bbox_vec,axis=0) / len(match_bbox_vec)
    return avg_bboxs

def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight[i] * bbox[i]
    return out_bbox

def bayesian_fusion_multiclass(match_score_vec, pred_class):
    log_positive_scores = np.log(match_score_vec[:,pred_class])
    pos_score = None
    neg_score = np.zeros(match_score_vec.shape)
    cnt = 0
    for i in range(match_score_vec.shape[1]):
        if i == pred_class:
            pos_score = match_score_vec[:, i]
        else:
            neg_score[:,cnt] = match_score_vec[:, i]
            cnt += 1
    
    # Background probability
    neg_score[:,-1] = 1 - np.sum(match_score_vec, axis=1) 
    
    log_positive_scores = np.log(pos_score)
    log_negative_scores = np.log(neg_score)
    
    #log_negative_scores = np.log(np.delete(match_score_vec, pred_class, 1))
    fused_positive = np.exp(np.sum(log_positive_scores))
    fused_negative = np.sum(np.exp(np.sum(log_negative_scores, axis=0)))
    out = fused_positive / (fused_positive + fused_negative)
    return out
    
def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight[i] * bbox[i]
    return out_bbox

def prepare_data(info1, info2, info3=''):
    bbox1 = np.array(info1['bbox'])
    bbox2 = np.array(info2['bbox'])
    score1 = np.array(info1['score'])
    score2 = np.array(info2['score'])
    class1 = np.array(info1['class'])
    class2 = np.array(info2['class'])
    out_logits = {}
    out_logits['1'] = np.array(info1['class_logits'])
    out_logits['2'] = np.array(info2['class_logits'])
    out_bbox = np.concatenate((bbox1, bbox2), axis=0)
    out_score = np.concatenate((score1, score2), axis=0)
    out_class = np.concatenate((class1, class2), axis=0)

    if info3:
        bbox3 = np.array(info3['bbox'])
        score3 = np.array(info3['score'])
        class3 = np.array(info3['class'])
        out_logits['3'] = np.array(info3['class_logits'])
        out_bbox = np.concatenate((out_bbox, bbox3), axis=0)
        out_score = np.concatenate((out_score, score3), axis=0)
        out_class = np.concatenate((out_class, class3), axis=0)
    
    if 'prob' in info1.keys():
        prob1 = np.array(info1['prob'])
        prob2 = np.array(info2['prob'])  
        out_prob = np.concatenate((prob1, prob2), axis=0)
        if info3:
            prob3 = np.array(info3['prob'])  
            out_prob = np.concatenate((out_prob, prob3), axis=0)
        return out_bbox, out_score, out_class, out_logits, out_prob
    else:    
        return out_bbox, out_score, out_class, out_logits

def prepare_data_gt(info1, info2, info_gt='', info3=''):
    bbox1 = np.array(info1['bbox'])
    bbox2 = np.array(info2['bbox'])
    
    score1 = np.array(info1['score'])
    score2 = np.array(info2['score'])
    class1 = np.array(info1['class'])
    class2 = np.array(info2['class'])
    
    out_logits1 = np.array(info1['class_logits'])
    out_logits2 = np.array(info2['class_logits'])
    out_logits = np.concatenate((out_logits1, out_logits2), axis=0)
    
    out_bbox = np.concatenate((bbox1, bbox2), axis=0)
    
    out_score = np.concatenate((score1, score2), axis=0)
    
    out_class = np.concatenate((class1, class2), axis=0)
    
    num_det = [len(class1), len(class2)]
    if info_gt:
        bbox_gt = np.array(info_gt['bbox'])
        out_bbox = np.concatenate((out_bbox, bbox_gt), axis=0)
        class_gt = np.array(info_gt['class'])
        out_score = np.concatenate((out_score, np.ones(len(class_gt))), axis=0)
        out_class = np.concatenate((out_class, class_gt), axis=0)
        num_det.append(len(class_gt))

    if 'prob' in info1.keys():
        prob1 = np.array(info1['prob'])
        prob2 = np.array(info2['prob'])  
        out_prob = np.concatenate((prob1, prob2), axis=0)
        if info3:
            prob3 = np.array(info3['prob'])  
            out_prob = np.concatenate((out_prob, prob3), axis=0)
        return out_bbox, out_score, out_class, out_logits, out_prob, num_det
    else:    
        return out_bbox, out_score, out_class, out_logits, num_det

def prepare_data_gt_1_det(info1, info_gt='', info3=''):
    bbox1 = np.array(info1['bbox'])
    out_score = np.array(info1['score'])
    out_class = np.array(info1['class'])
    out_logits = np.array(info1['class_logits'])
    out_bbox = np.array(bbox1)
    num_det = [len(out_class)]
    if info_gt:
        bbox_gt = np.array(info_gt['bbox'])
        out_bbox = np.concatenate((out_bbox, bbox_gt), axis=0)
        class_gt = np.array(info_gt['class'])
        out_class = np.concatenate((out_class, class_gt), axis=0)
        out_score = np.concatenate((out_score, np.ones(len(class_gt))), axis=0)
        num_det.append(len(class_gt))
    if 'prob' in info1.keys():
        out_prob = np.array(info1['prob'])
        return out_bbox, out_score, out_class, out_logits, out_prob, num_det
    else:    
        return out_bbox, out_score, out_class, out_logits, num_det

def determine_model(num_det, det_id):
    model_id_list = np.zeros(len(det_id), dtype=int)
    for i in range(len(det_id)):
        if det_id[i] < num_det[0]:
            model_id_list[i] = 0
        else:
            model_id_list[i] = 1
    return model_id_list

def nms_multiple_box(dets, scores, classes, logits, thresh, num_det, method):
    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    #scores = scores#dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    match_logits = np.zeros((1,8))
    match_class = np.zeros(1)
    out_boxes = np.zeros((1, 4))
    match_bboxs = []
    match_id = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #print(order)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Unmatched index in order list
        inds = np.where(ovr <= thresh)[0]
        # Matched index in order list
        match = np.where(ovr > thresh)[0]
        # Matched index in original list
        match_ind = order[match+1]

        match_score = list(scores[match_ind])
        match_bbox = list(dets[match_ind][:,:4])
        original_score = scores[i].tolist()
        original_bbox = dets[i][:4]
        
        if len(match_score) > 0:
            in_det_range = np.sum(num_det[:-1])
            # Assign matched scores
            #match_ind = match_ind.tolist()
            #match_ind.append(i)
            model_id_list = determine_model(num_det, match_ind)
            temp_score = np.zeros((1,8))
            
            for k in range(len(match_ind)):
                if not match_ind[k] in range(in_det_range):
                    #print('Its gt! Skip it...')
                    #pdb.set_trace()
                    continue                
                temp_score[0, model_id_list[k]*4:(model_id_list[k]+1)*4] = logits[match_ind[k]]
             
            match_logits = np.concatenate((match_logits, temp_score))
            
            # Cuurent box is groundtruth box
            #match_class = np.concatenate((match_class, [classes[i]]))
            
            if not i in range(in_det_range):         
                match_class = np.concatenate((match_class, [classes[i]]))
            else:# Cuurent box is not groundtruth box
                match_class = np.concatenate((match_class, [3]))
                
            # # # # # # # # # # # # # # #
            # Aggregate matched boxes
            # # # # # # # # # # # # # # #
            # Current box is not groundtruth
            if i in range(num_det[0] + num_det[1]):
                match_bbox += [original_bbox]
                match_score.append(original_score)

            if method == 'wt_box_fusion':
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            else:
                final_bbox = avg_bbox_fusion(match_bbox)
            match_bboxs.append(final_bbox)
            # Matched groundtruth ID
            match_id.append(match_ind)
           

            if i in range(in_det_range):
                # For current box
                temp_score = np.zeros((1,8))
                model_id_list = determine_model(num_det, [i])
                temp_score[0, model_id_list[0]*4:(model_id_list[0]+1)*4] = logits[i]
                match_logits = np.concatenate((match_logits, temp_score))
                match_class = np.concatenate((match_class, [3]))
                match_bboxs.append(original_bbox)
            #pdb.set_trace()
        # No matched bbox
        else:
            gt_start_id = np.sum(num_det[:-1])
            gt_end_id = gt_start_id + num_det[-1]
            # If current bbox is groundtruth and has no matched bbox
            if i in range(gt_start_id, gt_end_id):                
                order = order[inds + 1]                
                continue
            # If current bbox is false positive (box detected but should be background)
            else:
                # Save out the logits scores
                model_id = determine_model(num_det, [i])
                temp_score = np.zeros((1,8))                
                temp_score[0, model_id[0]*4:(model_id[0]+1)*4] = logits[i]                
                match_logits = np.concatenate((match_logits, temp_score))                
                # Assign background label
                match_class = np.concatenate((match_class, [3]))
                # Debug
                match_id.append(i)
                # Output bbox
                match_bboxs.append(original_bbox)
               
                #pdb.set_trace()
        #pdb.set_trace()
        # inds + 1 to reverse to original index
        order = order[inds + 1] 
    
    match_logits = match_logits[1:]
    match_class = match_class[1:]

    assert len(match_bboxs)==len(match_logits)
    assert len(match_class)==len(match_bboxs)

    match_bboxs = match_bboxs
    match_logits = torch.Tensor(match_logits)
    match_classes = torch.Tensor(match_class)

    return match_logits, match_classes, match_bboxs

def match_gt_box(in_boxes, in_classes, thresh):
    x1 = in_boxes[:, 0] #+ in_classes * 640
    y1 = in_boxes[:, 1] #+ in_classes * 512
    x2 = in_boxes[:, 2] #+ in_classes * 640
    y2 = in_boxes[:, 3] #+ in_classes * 512

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    xx1 = np.maximum(x1[0], x1[1:])
    yy1 = np.maximum(y1[0], y1[1:])
    xx2 = np.minimum(x2[0], x2[1:])
    yy2 = np.minimum(y2[0], y2[1:])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[0] + areas[1:] - inter)
    
    # Unmatched index in order list
    inds = np.where(ovr <= thresh)[0]
    # Matched index in order list
    match_id = np.where(ovr > thresh)[0]
    return match_id

def match_gt_box_percent(in_boxes, in_classes, thresh):
    x1 = in_boxes[:, 0] + in_classes * 640
    y1 = in_boxes[:, 1] + in_classes * 512
    x2 = in_boxes[:, 2] + in_classes * 640
    y2 = in_boxes[:, 3] + in_classes * 512

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    xx1 = np.maximum(x1[0], x1[1:])
    yy1 = np.maximum(y1[0], y1[1:])
    xx2 = np.minimum(x2[0], x2[1:])
    yy2 = np.minimum(y2[0], y2[1:])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[0] + areas[1:] - inter)
    
    # Unmatched index in order list
    inds = np.where(ovr <= thresh)[0]
    # Matched index in order list
    match_id = np.where(ovr > thresh)[0]
    return ovr

def nms_multiple_box_train(dets, scores, classes, logits, thresh, num_det, method):
    in_det_range = np.sum(num_det[:-1])
    gts = dets[in_det_range:]
    dets = dets[:in_det_range]
    scores = scores[:in_det_range]
    gt_class = classes[in_det_range:]
    classes = classes[:in_det_range]

    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    match_logits = np.zeros((1,8))
    match_class = np.zeros(1)
    out_boxes = np.zeros((1, 4))
    match_bboxs = []
    match_id = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #print(order)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Unmatched index in order list
        inds = np.where(ovr <= thresh)[0]
        # Matched index in order list
        match = np.where(ovr > thresh)[0]
        # Matched index in original list
        match_ind = order[match+1]

        match_score = list(scores[match_ind])
        match_bbox = dets[match_ind]
        original_score = scores[i].tolist()
        original_bbox = dets[i][:4]

        if len(match_score) > 0:
            # Get groundtruth labels
            match_ind = np.concatenate((match_ind, [i]))
            match_bbox = np.concatenate(match_bbox, axis=0)
            match_bbox = np.concatenate((match_bbox,original_bbox), axis=0)
            match_score += [original_score]

            in_classes = classes[match_ind[0]]
            in_classes = np.concatenate(([in_classes], gt_class), axis=0)
            in_boxes = dets[i]
            in_boxes = np.concatenate(([in_boxes], gts), axis=0)
            """
            in_scores = np.ones(len(in_boxes))*0.5
            in_scores[0] = 1
            keep_id = box_ops.batched_nms(torch.Tensor(in_boxes), torch.Tensor(in_scores), torch.Tensor(in_classes), 0.5)
            gt_id = match_gt_box(in_boxes, in_classes, 0.5)
            gt_id_nms = None
            for loop_i in range(len(in_boxes)):
                find_i = np.where(keep_id == loop_i)
                if len(find_i[0]) == 0:
                    print('gt id:', gt_id)
                    print('match: ',loop_i-1)
                    gt_id_nms = loop_i
            pdb.set_trace()
            """
            gt_id = match_gt_box(in_boxes, in_classes, 0.5)
            #pdb.set_trace()
            
            if len(gt_id) > 0: # Match groundtruth bbox
                if len(gt_id) > 1:
                    match_class = np.concatenate((match_class, [gt_class[gt_id[0]]]))
                else:
                    match_class = np.concatenate((match_class, gt_class[gt_id]))
            else: # Background class
                #order = order[inds+1]
                #continue
                #percent = match_gt_box_percent(in_boxes, in_classes, 0.5)
                #pdb.set_trace()            
                match_class = np.concatenate((match_class, [3]))
            
            # Get logits scores
            model_id_list = determine_model(num_det, match_ind)
            temp_score = np.zeros((1,8))
            for k in range(min(len(match_ind),2)):
                temp_score[0, model_id_list[k]*4:(model_id_list[k]+1)*4] = logits[match_ind[k]]            
            match_logits = np.concatenate((match_logits, temp_score))
        # No matched bbox
        else:
            # Check if match with groundtruth
            in_classes = classes[i]
            in_boxes = np.concatenate(([dets[i]], gts), axis=0)
            gt_id = match_gt_box(in_boxes, in_classes, 0.5)
            if len(gt_id) > 0:
                if len(gt_id) > 1:
                    match_class = np.concatenate((match_class, [gt_class[gt_id[0]]]))
                else:
                    match_class = np.concatenate((match_class, gt_class[gt_id]))
            else: # Background class
                match_class = np.concatenate((match_class, [3]))
            
            # Get logits scores
            model_id_list = determine_model(num_det, [i])
            temp_score = np.zeros((1,8))            
            temp_score[0, model_id_list[0]*4:(model_id_list[0]+1)*4] = logits[i]
            match_logits = np.concatenate((match_logits, temp_score))
            #pdb.set_trace()
        #pdb.set_trace()
        # inds + 1 to reverse to original index
        order = order[inds + 1] 
    
    match_logits = match_logits[1:]
    match_class = match_class[1:]
    
    assert len(match_class)==len(match_logits)

    match_logits = torch.Tensor(match_logits)
    match_classes = torch.Tensor(match_class)

    return match_logits, match_classes

def nms_multiple_box_eval(dets, scores, classes, logits, probs, thresh, num_det, method):
    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    #scores = scores#dets[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    match_logits = np.zeros((1,8))
    out_boxes = np.zeros((1, 4))
    match_bboxs = []
    match_id = []
    out_classes = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #print(order)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Unmatched index in order list
        inds = np.where(ovr <= thresh)[0]
        # Matched index in order list
        match = np.where(ovr > thresh)[0]
        # Matched index in original list
        match_ind = order[match+1]

        match_score = list(scores[match_ind])
        match_bbox = list(dets[match_ind][:,:4])
        original_score = scores[i].tolist()
        original_bbox = dets[i][:4]
        
        if len(match_score) > 0:
            # Assign matched scores
            match_ind = match_ind.tolist()
            match_ind.append(i)
            model_id_list = determine_model(num_det, match_ind)
            temp_score = np.zeros((1,8))
            
            for k in range(len(match_ind)):             
                temp_score[0, model_id_list[k]*4:(model_id_list[k]+1)*4] = logits[match_ind[k]]
             
            match_logits = np.concatenate((match_logits, temp_score))
            match_bbox += [original_bbox]
            
            if method == 'wt_box_fusion':
                match_score.append(original_score)
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            else:
                final_bbox = avg_bbox_fusion(match_bbox)
            match_bboxs.append(final_bbox)
            # Matched groundtruth ID
            match_id.append(match_ind)        
           
        # No matched bbox
        else:
            # Save out the logits scores
            model_id = determine_model(num_det, [i])
            temp_score = np.zeros((1,8))                
            temp_score[0, model_id[0]*4:(model_id[0]+1)*4] = logits[i]                
            match_logits = np.concatenate((match_logits, temp_score))                

            # Debug
            match_id.append(i)
            # Output bbox
            match_bboxs.append(original_bbox)
            #pdb.set_trace()
        # inds + 1 to reverse to original index
        out_classes.append(classes[i])
        order = order[inds + 1] 
    
    match_logits = match_logits[1:]

    assert len(match_bboxs)==len(match_logits)
    assert len(out_classes)==len(match_logits)

    match_bboxs = match_bboxs
    match_logits = torch.Tensor(match_logits)
  
    return match_logits, match_bboxs, out_classes

def evaluate(cfg, evaluator, det_1, det_2, predictor, method, bayesian=False):
    evaluator.reset()
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    num_img = len(det_2['image'])
    count_1=0
    count_2=0
    count_fusion=0

    print('Method: ', method)

    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    num_img = len(det_2['image'])
    count_1=0
    count_2=0
    count_fusion=0
    X = None
    Y = np.array([])
    cnt = 0

    for i in range(num_img):
        info_1 = {}
        info_1['img_name'] = det_1['image'][i]
        info_1['bbox'] = det_1['boxes'][i]
        info_1['score'] = det_1['scores'][i]
        info_1['class'] = det_1['classes'][i]
        info_1['class_logits'] = det_1['class_logits'][i]
        if 'probs' in det_1.keys():
            info_1['prob'] = det_1['probs'][i]

        info_2 = {}
        info_2['img_name'] = det_2['image'][i].split('.')[0] + '.jpeg'
        info_2['bbox'] = det_2['boxes'][i]
        info_2['score'] = det_2['scores'][i]
        info_2['class'] = det_2['classes'][i]
        info_2['class_logits'] = det_2['class_logits'][i]
        if 'probs' in det_2.keys():
            info_2['prob'] = det_2['probs'][i]
        
        #img_id = int(info_1['img_name'].split('.')[0].split('_')[1]) - 1
        img_id = det_1['image_id'][i]
         
        # If no any detection in two results
        if len(info_1['bbox']) == 0 and len(info_2['bbox']) == 0:
            continue
        # If no detection in 1st model:
        elif len(info_1['bbox']) == 0:
            print('model 1 miss detected')
            in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_1_det(info_2)
        elif len(info_2['bbox']) == 0:
            print('model 2 miss detected')
            in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_1_det(info_1)
        else:
            in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt(info_1, info_2)
        score_results, box_results, class_results = nms_multiple_box_eval(in_boxes, in_scores, in_class, in_logits, in_prob, 0.5, num_det, method)

        if bayesian:
            # summing logits
            sum_logits = score_results[:,:4] + score_results[:,4:]
            pred_prob_multiclass = F.softmax(torch.Tensor(sum_logits)).tolist()
            out_scores = np.max(pred_prob_multiclass, axis=1)
            out_class = np.argmax(pred_prob_multiclass, axis=1)
        else:            
            pred_prob_multiclass = predictor.predict_proba(score_results)
            out_scores = np.max(pred_prob_multiclass, axis=1)
            out_class = np.argmax(pred_prob_multiclass, axis=1)

        #pdb.set_trace()
        """
        Send information to evaluator
        """
        # Image info
        file_name = img_folder + info_1['img_name'].split('.')[0] + '.jpeg'
        img = cv2.imread(file_name)
        H, W, _ = img.shape

        # Handle inputs
        inputs = []
        input_info = {}
        input_info['file_name'] = file_name
        input_info['height'] = H
        input_info['width'] = W
        input_info['image_id'] = det_1['image_id'][i]
        input_info['image'] = torch.Tensor(img)
        inputs.append(input_info)
        
        # Handle outputs
        outputs = []
        out_info = {}
        proposals = Instances([H, W])
        proposals.pred_boxes = Boxes(box_results)
        proposals.scores = torch.Tensor(out_scores)
        proposals.pred_classes = torch.Tensor(out_class)
        out_info['instances'] = proposals
        outputs.append(out_info)
        evaluator.process(inputs, outputs)
        
    results = evaluator.evaluate(out_eval_path='FLIR_pooling_.out')
    
    if results is None:
        results = {}
    
    avgRGB = count_1 / num_img
    avgThermal = count_2 / num_img
    avgNMS = count_fusion / num_img

    print('Avg bbox for RGB:', avgRGB, "average count thermal:", avgThermal, 'average count nms:', avgNMS)
    return results

def train_late_fusion(det_1, det_2, anno):
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    num_img = len(det_2['image'])
    count_1=0
    count_2=0
    count_fusion=0
    method = 'train'
    X = None
    Y = np.array([])
    cnt = 0
    cnt_inst = 0

    for i in range(num_img):
        info_1 = {}
        info_1['img_name'] = det_1['image'][i]
        info_1['bbox'] = det_1['boxes'][i]
        info_1['score'] = det_1['scores'][i]
        info_1['class'] = det_1['classes'][i]
        info_1['class_logits'] = det_1['class_logits'][i]
        if 'probs' in det_1.keys():
            info_1['prob'] = det_1['probs'][i]

        info_2 = {}
        info_2['img_name'] = det_2['image'][i].split('.')[0] + '.jpeg'
        info_2['bbox'] = det_2['boxes'][i]
        info_2['score'] = det_2['scores'][i]
        info_2['class'] = det_2['classes'][i]
        info_2['class_logits'] = det_2['class_logits'][i]
        if 'probs' in det_2.keys():
            info_2['prob'] = det_2['probs'][i]
        
        #img_id = int(info_1['img_name'].split('.')[0].split('_')[1]) - 1
        img_id = det_1['image_id'][i]
        box_gt = []
        class_gt = []
        info_gt = {}
        
        #print('img_id:',img_id)
        if img_id in anno.keys():
            # Handle groundtruth
            anno_gt = anno[img_id]
            for j in range(len(anno_gt)):
                box = anno_gt[j]['bbox']
                box_gt.append([box[0], box[1], box[0]+box[2], box[1]+box[3]])
                class_gt.append(anno_gt[j]['category_id'])
            info_gt['bbox'] = box_gt
            info_gt['class'] = class_gt
            
            # If no any detection in two results
            if len(info_1['bbox']) == 0 and len(info_2['bbox']) == 0:
                continue
            # If no detection in 1st model:
            elif len(info_1['bbox']) == 0:
                #print('model 1 missing detection')
                in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_1_det(info_2, info_gt=info_gt)
                score_results, class_results = nms_multiple_box_train(in_boxes, in_scores, in_class, in_logits, 0.5, num_det, method)
            elif len(info_2['bbox']) == 0:
                #print('model 2 missing detection')
                in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_1_det(info_1, info_gt=info_gt)
                score_results, class_results = nms_multiple_box_train(in_boxes, in_scores, in_class, in_logits, 0.5, num_det, method)
            else:
                in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt(info_1, info_2, info_gt=info_gt)                
                score_results, class_results = nms_multiple_box_train(in_boxes, in_scores, in_class, in_logits, 0.5, num_det, method)
            
            cnt_inst += len(class_results)
            if len(score_results):
                if cnt==0:
                    X = score_results
                else:
                    try:
                        X = np.concatenate((X, score_results))
                    except:
                        pdb.set_trace()
                Y = np.concatenate((Y, class_results))
                cnt += 1
        else:
            continue
    print('total training instanse: ', cnt_inst)
    return X, Y

if __name__ == '__main__':

    """
    Handle training dataset annotations
    """
    data_set = 'train'
    in_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_RGBT_pairs_3_class.json'#thermal_annotations_4_channel_no_dogs.json'
    img_folder = '../../../Datasets/FLIR/'+data_set+'/RGB'
    data = json.load(open(in_anno_file, 'r'))
    info = data['info']
    categories = data['categories']
    licenses = data['licenses']
    annos = data['annotations']
    images = data['images']

    annos_on_img = {}
    img_cnt = 1
    anno_train_gt = {}
    img = []
    
    # Re-arrange groundtruth annotations: 1 image per element in list
    for i in range(len(annos)):
        if annos[i]['image_id'] == img_cnt:
            img.append(annos[i])
        else:
            try:
                anno_train_gt[img[0]['image_id']] = img
            except:
                pdb.set_trace()
            img = []
            img.append(annos[i])
            img_cnt += 1
    anno_train_gt[img[0]['image_id']] = img

    """
    Handle validation dataset annotations
    """
    data_set = 'val'
    in_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_RGBT_pairs_3_class.json'#thermal_annotations_4_channel_no_dogs.json'
    img_folder = '../../../Datasets/FLIR/'+data_set+'/RGB'
    data = json.load(open(in_anno_file, 'r'))
    info = data['info']
    categories = data['categories']
    licenses = data['licenses']
    annos = data['annotations']
    images = data['images']

    img_dict = {}
    for i in range(len(images)):
        img = images[i]
        img_id = img['id']
        file_name = img['file_name']
        file_name_id = int(file_name.split('.')[0].split('FLIR_')[1])
        img_dict[img_id] = file_name_id

    annos_on_img = {}
    img_cnt = 0
    anno_val_gt = {}
    img = []
    
    # Re-arrange groundtruth annotations: 1 image per element in list
    for i in range(len(annos)):
        # The same image
        if annos[i]['image_id'] == img_cnt:
            img.append(annos[i])
        # Loop to new image, should move to next image
        else:
            # Get img id from the existing img list, and set as dictionary key
            anno_val_gt[img[0]['image_id']] = img
            # Record new image info
            img = []
            img.append(annos[i])
            img_cnt += 1
    anno_val_gt[img[0]['image_id']] = img

    data_set = 'val'
    data_folder = 'out/box_predictions/3_class/'
    dataset = 'FLIR'
    IOU = 50                 
    time = 'all'
    model_1 = 'early_fusion'
    model_2 = 'mid_fusion'
    model_3 = 'thermal_only'
    if time == 'Day':
        val_file_name = 'thermal_annotations_4_channel_no_dogs_Day.json'#'RGB_annotations_4_channel_no_dogs.json'#'thermal_annotations_4_channel_no_dogs_Day.json'#
        det_file_1 = data_folder + 'train_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_2 = data_folder + 'train_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_3 = data_folder + 'train_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'

        val_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        val_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        val_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        
    elif time == 'Night':
        val_file_name = 'thermal_annotations_4_channel_no_dogs_Night.json'
        det_file_1 = data_folder + 'train_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_2 = data_folder + 'train_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_3 = data_folder + 'train_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'

        val_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        val_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        val_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        
    else:
        """
        3 class with multiclass probability score
        """
        # Training data
        det_file_1 = data_folder + 'train_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_2 = data_folder + 'train_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_3 = data_folder + 'train_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        
        # Validation data
        val_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        val_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        val_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        
        val_file_name = 'thermal_RGBT_pairs_3_class.json'#'thermal_annotations_4_channel_no_dogs.json'
    
    print('detection file 1:', val_file_1)
    print('detection file 2:', val_file_2)
    print('detection file 3:', val_file_3)
    
    path_1 = '../../../Datasets/FLIR/' + data_set + '/resized_RGB/'
    path_2 = '../../../Datasets/FLIR/' + data_set + '/thermal_8_bit/'
    out_folder = 'out/box_comparison/'
    #train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4_channel_no_dogs.json'
    
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
    det_3 = json.load(open(det_file_3, 'r'))
    val_1 = json.load(open(val_file_1, 'r'))
    val_2 = json.load(open(val_file_2, 'r'))
    #val_3 = json.load(open(det_file_3, 'r'))
    evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='out/mAP/FLIR_Baysian_Day.out')
    method = 'avg_score'#'baysian_wt_score_box'#'sumLogits_softmax'#'avgLogits_softmax'#'baysian_avg_bbox'#'avg_score'#'pooling' #'baysian'#'nms'
    #result = apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, det_3, method)
    save_file_name = 'train_labels_2_model.npz'#'train_labels_2_model.npz'#'train_labels_2_model_0219.npz'#'train_labels_2_model_w_false_positives_new.npz'#'train_labels_2_model_new.npz'
    #save_file_name = 'train_labels_2_model_rm_false_positives.npz'
    """
    # Get training labels
    print('Perpare training data ... ')
    X_train, Y_train = train_late_fusion(det_1, det_2, anno_train_gt)
    np.savez(save_file_name, X=X_train, Y=Y_train)
    #pdb.set_trace()
    """
    print('Loading saved data ...')
    train_data = np.load(save_file_name)
    X_train = train_data['X']
    Y_train = train_data['Y']
    print('Perpare validation data ... ')
    X_val, Y_val = train_late_fusion(val_1, val_2, anno_val_gt)
    print('Done data preparation.')
    pdb.set_trace()
    
    from sklearn.linear_model import LogisticRegression
    train_samples = len(Y_train)

    #save_file_name = 'learning_late_fusion_weight_fuse_2_models_l1_w_bias.npz'
    ##########################
    # User Setting
    ##########################
    penalty = 'l2'
    use_bias = False
    method = 'wt_box_fusion' #'avg_box_fusion' #'wt_box_fusion'
    folder = './'#'out/models/'
    ##########################
    if use_bias:
        save_model_name = folder + 'learned_late_fusion_2_models_'+penalty+'_w_bias.sav'
    else:
        save_model_name = folder + 'learned_late_fusion_2_models_'+penalty+'_w_o_bias.sav'
    
    #save_model_name = foler + 'learned_late_fusion_2_models_w_o_.sav'
    """
    # Predict
    predictor = pickle.load(open(save_model_name, 'rb'))
    result = evaluate(cfg, evaluator, val_1, val_2, anno_val_gt, predictor, method)
    pdb.set_trace()
    """
    max_score = 0
    loop_time = 1
    #class_weight = {0:1.0, 1:3.2, 2:0.7, 3:0.095}
    #class_weight = {0:1.0, 1:3.0, 2:0.8, 3:0.095}
    #class_weight = {0:0.61,1:0.97,2:0.5,3:0.15}
    class_weight = {0:0.61, 1:0.97, 2:0.6, 3:0.15}
    for i in range(loop_time):
        print('------- iteration: ', i, ' -------')
        print('Penalty: ', penalty)
        if use_bias: print('Using bias ...')        
        else: print('Not using bias ...')
        #predictor = LogisticRegression(C=100. / train_samples, penalty=penalty, solver='saga', tol=0.1, max_iter=200, fit_intercept=use_bias)
        predictor = LogisticRegression(C=20, penalty=penalty, solver='saga',verbose=1, tol=0.0001, max_iter=200, fit_intercept=use_bias, class_weight=class_weight)
        predictor.fit(X_train, Y_train)            
        result = evaluate(cfg, evaluator, val_1, val_2, predictor, method)
        AP50_score = result['bbox']['AP50']
                
        if AP50_score > max_score:
            max_score = AP50_score
            out_weight = predictor.coef_
            param = predictor.get_params()
            bias = predictor.intercept_
            print('Saving model ...')
            pickle.dump(predictor, open(save_model_name, 'wb'))
            #pdb.set_trace()
            #np.savez(save_file_name, weight=out_weight, params=param, bias=bias)
        print('--------------------------')
        print('current highest mAP:', max_score)
    print('max AP50:', max_score)
    pdb.set_trace()
    