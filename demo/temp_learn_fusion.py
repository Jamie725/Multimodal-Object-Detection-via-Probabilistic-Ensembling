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
    
def nms_1(info_1, info_2, info_3=''):
    # Boxes
    boxes = info_1['bbox'].copy()
    boxes.extend(info_2['bbox'])
    # Scores
    scores = info_1['score'].copy()
    scores.extend(info_2['score'])
    # Classes
    classes = info_1['class'].copy()
    classes.extend(info_2['class'])
    if info_3:
        boxes.extend(info_3['bbox'])
        scores.extend(info_3['score'])
        classes.extend(info_3['class'])
    
    classes = torch.Tensor(classes)
    scores = torch.Tensor(scores)
    boxes = torch.Tensor(boxes)
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
    num_det = [len(class1), len(class2)]

    if info3:
        bbox3 = np.array(info3['bbox'])
        score3 = np.array(info3['score'])
        class3 = np.array(info3['class'])
        out_logits['3'] = np.array(info3['class_logits'])
        out_bbox = np.concatenate((out_bbox, bbox3), axis=0)
        out_score = np.concatenate((out_score, score3), axis=0)
        out_class = np.concatenate((out_class, class3), axis=0)
        num_det.append(len(class3))

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

def nms_bayesian(dets, scores, classes, probs, thresh, method):
    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    #scores = scores#dets[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    match_scores = []
    match_bboxs = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
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
        
        match_prob = list(probs[match_ind])
        match_score = list(scores[match_ind])
        match_bbox = list(dets[match_ind][:,:4])
        original_prob = probs[i]
        original_score = scores[i].tolist()
        original_bbox = dets[i][:4]
        if len(match_score)>0:
            match_score += [original_score]
            match_prob += [original_prob]
            #pdb.set_trace()
            if method == 'avg_score':
                final_score = np.mean(np.asarray(match_score))
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'bayesian':
                #final_score = bayesian_fusion_multiclass(np.asarray(match_score))
                final_score = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'bayesian_avg_bbox':
                final_score = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                #final_score = bayesian_fusion_multiclass(np.asarray(match_score))
                match_bbox += [original_bbox]
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'bayesian_wt_score_box':
                final_score = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                #final_score = bayesian_fusion(np.asarray(match_score))
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            else:
                print('method: ', method, ' not found!')
            
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

def handle_logits(logits, classes):
    logits1 = logits['1']
    logits2 = logits['2']
    out_logits = np.concatenate((logits1, logits2), axis=0)
    if '3' in logits.keys():
        logits3 = logits['3']
        out_logits = np.concatenate((out_logits, logits3), axis=0)
    return out_logits

def determine_model(num_det, det_id):
    model_id_list = np.zeros(len(det_id), dtype=int)
    for i in range(len(det_id)):
        if det_id[i] < num_det[0]:
            model_id_list[i] = 0
        else:
            model_id_list[i] = 1
    return model_id_list

def nms_logits(dets, scores, classes, logits, thresh, method, predictor='', num_det=''):
    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    
    out_logits = handle_logits(logits, classes)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order = out_max_logits.argsort()[::-1]
    order = scores.argsort()[::-1]
    keep = []
    match_scores = []
    match_bboxs = []
    final_logits = np.zeros((1,8))
    logits_softmax = []
    match_case = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
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

        match_score = list(out_logits[match_ind])
        match_bbox = list(dets[match_ind][:,:4])
        original_score = out_logits[i].tolist()        
        original_bbox = dets[i][:4]

        if len(match_score)>0:
            match_ind = match_ind.tolist()
            match_ind.append(i)
            match_score += [original_score]
            match_prob = scores[match_ind]
            if method == 'avgLogits_softmax':
                final_score = np.mean(np.asarray(match_score), axis=0)
                final_score = F.softmax(torch.Tensor(final_score), dim=0)[classes[i]].tolist()
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'sumLogits_softmax':
                final_score = np.sum(np.asarray(match_score), axis=0)
                final_score = F.softmax(torch.Tensor(final_score), dim=0)[classes[i]].tolist()
                final_bbox = avg_bbox_fusion(match_bbox)
                pdb.set_trace()
            elif method == 'learned_fusion':
                # SumLogits
                final_score = np.sum(np.asarray(match_score), axis=0)
                final_score = F.softmax(torch.Tensor(final_score), dim=0)[classes[i]].tolist()
                # Prepare out logits list
                model_id_list = determine_model(num_det, match_ind)
                temp_score = np.zeros((1,8))
                temp_logits = np.zeros((num_det[0] + num_det[1], 4))
                temp_logits[:num_det[0],:] = logits['1']
                temp_logits[num_det[0]:,:] = logits['2']
                #final_bbox = avg_bbox_fusion(match_bbox)
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_prob)
                #pdb.set_trace()
                for k in range(len(match_ind)):            
                    temp_score[0, model_id_list[k]*4:(model_id_list[k]+1)*4] += temp_logits[match_ind[k]]
                final_logits = np.concatenate((final_logits, temp_score))

                sum_score = temp_score[0,:4] + temp_score[0,4:]
                sum_score_softmax = F.softmax(torch.Tensor(sum_score), dim=0)[classes[i]].tolist()
                logits_softmax.append(sum_score_softmax)
                match_case.append(0)
                #pdb.set_trace()
            match_scores.append(final_score)            
            match_bboxs.append(final_bbox)
            
        else:
            final_score = F.softmax(torch.Tensor(original_score), dim=0)[classes[i]].tolist()
            match_scores.append(final_score)
            if method == 'learned_fusion':
                temp_logits = np.zeros((num_det[0] + num_det[1], 4))
                temp_logits[:num_det[0],:] = logits['1']
                temp_logits[num_det[0]:,:] = logits['2']
                temp_score = np.zeros((1,8))
                model_id = determine_model(num_det, [i])
                temp_score[0, model_id[0]*4:(model_id[0]+1)*4] = temp_logits[i]
                final_logits = np.concatenate((final_logits, temp_score))
                sum_score = temp_score[0,:4] + temp_score[0,4:]
                sum_score_softmax = F.softmax(torch.Tensor(sum_score), dim=0)[classes[i]].tolist()
                logits_softmax.append(sum_score_softmax)
                match_case.append(1)
            #pdb.set_trace()
            match_bboxs.append(original_bbox)
            
        #del temp_logits
        #pdb.set_trace()
        #print(match_scores)
        order = order[inds + 1]

    
    #pdb.set_trace()
    assert len(keep)==len(match_scores)
    assert len(keep)==len(match_bboxs)

    match_bboxs = match_bboxs
    match_scores = torch.Tensor(match_scores)
    match_classes = torch.Tensor(classes[keep])
    if method == 'learned_fusion':
        final_logits = final_logits[1:, :]
        assert len(keep) == len(final_logits)
        return keep,match_scores,match_bboxs, match_classes, final_logits, logits_softmax, match_case
    else:
        return keep,match_scores,match_bboxs, match_classes
def fusion(method, info_1, info_2, info_3='', predictor=''):
    threshold = 0.5
    if method == 'nms':            
        out_boxes, out_scores, out_class = nms_1(info_1, info_2, info_3=info_3)
        #in_boxes, in_scores, in_class, in_logits, in_prob = prepare_data(info_1, info_2, info3=info_3)
    elif method == 'pooling':
        #in_boxes, in_scores, in_class = prepare_data(info_1, info_2, info3=info_3)
        in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data(info_1, info_2, info3=info_3)
        out_boxes = in_boxes
        out_scores = torch.Tensor(in_scores)
        out_class = torch.Tensor(in_class)
    elif method == 'bayesian' or method == 'bayesian_avg_bbox' or method == 'avg_score' or method == 'bayesian_wt_score_box':
        #in_boxes, in_scores, in_class = prepare_data(info_1, info_2, info3=info_3)
        in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data(info_1, info_2, info3=info_3)
        #keep, out_scores, out_boxes, out_class = nms_bayesian(in_boxes, in_scores, in_class, threshold, method)
        keep, out_scores, out_boxes, out_class = nms_bayesian(in_boxes, in_scores, in_class, in_prob, threshold, method)        
    elif method == 'avgLogits_softmax' or method == 'sumLogits_softmax':
        in_boxes, in_scores, in_class, in_logits, _, num_det = prepare_data(info_1, info_2, info3=info_3)
        keep, out_scores, out_boxes, out_class = nms_logits(in_boxes, in_scores, in_class, in_logits, threshold, method)
    elif method == 'learned_fusion':
        in_boxes, in_scores, in_class, in_logits, _, num_det = prepare_data(info_1, info_2, info3=info_3)
        keep, out_scores, out_boxes, out_class, out_logits, logits_score, match_case = nms_logits(in_boxes, in_scores, in_class, in_logits, threshold, method, num_det=num_det, predictor=predictor)
        return out_boxes, out_scores, out_class, out_logits, logits_score, match_case
    elif method == 'train':
        in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt(info_1, info_2, info_gt=info_gt)
        keep, out_scores, out_boxes, out_class, out_logits = nms_logits(in_boxes, in_scores, in_class, in_logits, threshold, method, num_det=num_det)
    else:
        print('No matched method:', mehtod,' found!')
    return out_boxes, out_scores, out_class
    
def draw_box(img, bbox, color):
    for i in range(len(bbox)):
        img = cv2.rectangle(img,  (int(bbox[i][0]+0.5), int(bbox[i][1]+0.5)),  (int(bbox[i][2]+0.5), int(bbox[i][3]+0.5)), color, 2)
    return img

def match_gt_box(in_boxes, thresh):
    x1 = in_boxes[:, 0] 
    y1 = in_boxes[:, 1] 
    x2 = in_boxes[:, 2] 
    y2 = in_boxes[:, 3] 

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
def prepare_data_gt_orig(info1, info2, info_gt='', info3=''):
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

def prepare_data_gt(info1, info2, info_gt='', info3=''):
    bbox1 = np.array(info1['bbox'])
    bbox2 = np.array(info2['bbox'])
    
    score1 = np.array(info1['score'])
    score2 = np.array(info2['score'])
    class1 = np.array(info1['class'])
    class2 = np.array(info2['class'])
    """
    out_logits1 = np.array(info1['class_logits'])
    out_logits2 = np.array(info2['class_logits'])
    out_logits = np.concatenate((out_logits1, out_logits2), axis=0)
    """
    out_logits = {}
    out_logits['1'] = np.array(info1['class_logits'])
    out_logits['2'] = np.array(info2['class_logits'])
    
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

def get_train_labels(det_1, det_2, anno, det_3=''):
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'    
    num_img = len(det_2['image'])
    out_Y = []
    out_X = np.zeros((1,8))

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
        
        if len(info_1['bbox']) > 0:
            num_1 = 1
        else:
            num_1 = 0
        if len(info_2['bbox']) > 0:
            num_2 = 1
        else:
            num_2 = 0
        
        num_detections = num_1 + num_2

        if det_3:
            info_3 = {}
            info_3['img_name'] = det_3['image'][i].split('.')[0] + '.jpeg'
            info_3['bbox'] = det_3['boxes'][i]
            info_3['score'] = det_3['scores'][i]
            info_3['class'] = det_3['classes'][i]
            info_3['class_logits'] = det_3['class_logits'][i]
            if 'probs' in det_3.keys():
                info_3['prob'] = det_3['probs'][i]
            if len(info_3['bbox']) > 0:
                num_3 = 1
            else:
                num_3 = 0

            num_detections += num_3
        
        # No detections
        if num_detections == 0:
            continue
        # Only 1 model detection
        elif num_detections == 1:   
            if len(info_1['bbox']) > 0:
                out_boxes = np.array(info_1['bbox'])
                out_class = torch.Tensor(info_1['class'])
                out_scores = torch.Tensor(info_1['score'])
                num_det_1 = len(info_1['class_logits'])
                out_logits = np.zeros((num_det_1,8))
                for k in range(num_det_1):
                    out_logits[k, :4] = info_1['class_logits'][k]
                #pdb.set_trace()
            elif len(info_2['bbox']) > 0:
                out_boxes = np.array(info_2['bbox'])
                out_class = info_2['class']
                out_scores = info_2['score']                
                num_det_2 = len(info_2['class_logits'])
                out_logits = np.zeros((num_det_2,8))
                for k in range(num_det_2):
                    out_logits[k, 4:] = info_2['class_logits'][k]
            else:
                if det_3:
                    out_boxes = np.array(info_3['bbox'])
                    out_class = torch.Tensor(info_3['class'])
                    out_scores = torch.Tensor(info_3['score'])
        # Only two models with detections
        elif num_detections == 2:
            #pdb.set_trace()
            if not det_3:
                out_boxes, out_scores, out_class, out_logits = fusion(method, info_1, info_2)            
            else:
                if len(info_1['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_2, info_3)
                elif len(info_2['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_3)
                else:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_2)
        # All 3 models detected things
        else:
            out_boxes, out_scores, out_class = fusion(method, info_1, info_2, info_3=info_3)
        # Handle groundtruth
        img_id = det_1['image_id'][i]
        box_gt = []
        class_gt = []
        info_gt = {}
        print(i)
        if img_id in anno.keys():
            anno_gt = anno[img_id]
            for j in range(len(anno_gt)):
                box = anno_gt[j]['bbox']
                box_gt.append([box[0], box[1], box[0]+box[2], box[1]+box[3]])
                class_gt.append(anno_gt[j]['category_id'])
            info_gt['bbox'] = box_gt
            info_gt['class'] = class_gt
            
            img_id = det_1['image_id'][i]
            for k in range(len(out_boxes)):
                in_boxes = out_boxes[k]
                in_boxes = np.concatenate(([in_boxes], info_gt['bbox']), axis=0)
                match_id = match_gt_box(in_boxes, 0.5)
                if len(match_id) == 0:
                    out_Y.append(3)
                else:
                    out_Y.append(info_gt['class'][match_id[0]])
            out_X = np.concatenate((out_X, out_logits), axis=0) 
            assert(len(out_X)-1, len(out_Y))
            #pdb.set_trace()
                
        else:# False positives!
            out_X = np.concatenate((out_X, out_logits), axis=0)
            for k in range(len(out_boxes)):
                out_Y.append(3)
            
    out_X = out_X[1:, :]
    return out_X, out_Y

def apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, method, predictor, det_3='', bayesian=False):
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
        
        if len(info_1['bbox']) > 0:
            num_1 = 1
        else:
            num_1 = 0
        if len(info_2['bbox']) > 0:
            num_2 = 1
        else:
            num_2 = 0
        
        num_detections = num_1 + num_2

        if det_3:
            info_3 = {}
            info_3['img_name'] = det_3['image'][i].split('.')[0] + '.jpeg'
            info_3['bbox'] = det_3['boxes'][i]
            info_3['score'] = det_3['scores'][i]
            info_3['class'] = det_3['classes'][i]
            info_3['class_logits'] = det_3['class_logits'][i]
            if 'probs' in det_3.keys():
                info_3['prob'] = det_3['probs'][i]
            if len(info_3['bbox']) > 0:
                num_3 = 1
            else:
                num_3 = 0

            num_detections += num_3
        
        # No detections
        if num_detections == 0:
            continue
        # Only 1 model detection
        elif num_detections == 1:   
            if len(info_1['bbox']) > 0:
                out_boxes = np.array(info_1['bbox'])
                out_class = torch.Tensor(info_1['class'])
                out_scores = torch.Tensor(info_1['score'])
                num_det_1 = len(info_1['class_logits'])
                out_logits = np.zeros((num_det_1,8))
                for k in range(num_det_1):
                    out_logits[k, :4] = info_1['class_logits'][k]
                
            elif len(info_2['bbox']) > 0:
                out_boxes = np.array(info_2['bbox'])
                out_class = torch.Tensor(info_2['class'])
                out_scores = torch.Tensor(info_2['score'])
                num_det_2 = len(info_1['class_logits'])
                out_logits = np.zeros((num_det_2,8))
                for k in range(num_det_1):
                    out_logits[k, 4:] = info_1['class_logits'][k]
            else:
                if det_3:
                    out_boxes = np.array(info_3['bbox'])
                    out_class = torch.Tensor(info_3['class'])
                    out_scores = torch.Tensor(info_3['score'])
        # Only two models with detections
        elif num_detections == 2:
            #pdb.set_trace()
            if not det_3:
                if method == 'learned_fusion':
                    out_boxes, out_scores, out_class, out_logits, _, _ = fusion(method, info_1, info_2, predictor=predictor)
                else:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_2)
            else:
                if len(info_1['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_2, info_3)
                elif len(info_2['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_3)
                else:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_2)
        # All 3 models detected things
        else:
            out_boxes, out_scores, out_class = fusion(method, info_1, info_2, info_3=info_3)
        
        if bayesian:
            sum_logits = out_logits[:,:4] + out_logits[:,4:]
            pred_prob_multiclass = F.softmax(torch.Tensor(sum_logits)).tolist()
            out_scores = np.max(pred_prob_multiclass, axis=1)
            out_class = np.argmax(pred_prob_multiclass, axis=1)
        elif method == 'learned_fusion':
            pred_prob_multiclass = predictor.predict_proba(out_logits)
            #pred_prob_multiclass = predictor(torch.Tensor(out_logits).cuda(0))            
            out_scores = np.max(pred_prob_multiclass, axis=1)
            out_class = np.argmax(pred_prob_multiclass, axis=1)

        file_name = img_folder + info_1['img_name'].split('.')[0] + '.jpeg'
        img = cv2.imread(file_name)
        try:
            H, W, _ = img.shape
        except:
            pdb.set_trace()

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
        #pdb.set_trace()
        """
        img = draw_box(img, out_boxes, (0,255,0))
        out_img_name = 'out_img_baysian_fusion/' + file_name.split('thermal_8_bit/')[1].split('.')[0]+'_baysian_avg_bbox.jpg'
        cv2.imwrite(out_img_name, img)
        #pdb.set_trace()
        
        if '09115' in file_name:
            out_img_name = 'out_img_baysian_fusion/' + file_name.split('thermal_8_bit/')[1].split('.')[0]+'_baysian_avg_bbox.jpg'
            pdb.set_trace()
            cv2.imwrite(out_img_name, img)
        """
        
    results = evaluator.evaluate(out_eval_path='FLIR_pooling_.out')
    
    if results is None:
        results = {}

    return results

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
            in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_orig(info_1, info_2)
        score_results, box_results, class_results = nms_multiple_box_eval(in_boxes, in_scores, in_class, in_logits, in_prob, 0.5, num_det, method)

        
        if bayesian:
            # summing logits
            sum_logits = score_results[:,:4] + score_results[:,4:]
            pred_prob_multiclass = F.softmax(torch.Tensor(sum_logits)).tolist()
            out_scores = np.max(pred_prob_multiclass, axis=1)
            out_class = np.argmax(pred_prob_multiclass, axis=1)
        else:
            sum_logits = score_results[:,:4] + score_results[:,4:]
            pred_prob_multiclass = F.softmax(torch.Tensor(sum_logits)).tolist()
            #pdb.set_trace()
            #pred_prob_multiclass = predictor.predict_proba(score_results)
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
    
    return results

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

if __name__ == '__main__':
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
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
    elif time == 'Night':
        val_file_name = 'thermal_annotations_4_channel_no_dogs_Night.json'
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
    else:
        # more classes
        #det_file_1 = data_folder + 'val_mid_fusion_predictions_IOU50_with_logits.json'
        #det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_with_logits.json'
        #det_file_3 = data_folder + 'val_thermal_only_predictions_IOU50_with_logits.json'
        
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
    
    print('detection file 1:', det_file_1)
    print('detection file 2:', det_file_2)
    print('detection file 3:', det_file_3)
    
    path_1 = '../../../Datasets/FLIR/' + data_set + '/resized_RGB/'
    path_2 = '../../../Datasets/FLIR/' + data_set + '/thermal_8_bit/'
    out_folder = 'out/box_comparison/'
    #train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4_channel_no_dogs.json'
    
    val_json_path = '../../../Datasets/'+dataset+'/val/' + val_file_name
    val_folder = '../../../Datasets/FLIR/val/thermal_8_bit'

    val_1 = json.load(open(val_file_1, 'r'))
    val_2 = json.load(open(val_file_2, 'r'))

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
    evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='out/mAP/FLIR_Baysian_Day.out')
    method = 'learned_fusion'#'baysian_wt_score_box'#'sumLogits_softmax'#'avgLogits_softmax'#'baysian_avg_bbox'#'avg_score'#'pooling' #'baysian'#'nms'
    # 3 model input
    #result = apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, method, det_3=det_3)
    # Two model input
    """
    folder = './'#'learned_models/'
    save_model_name = folder + 'learned_late_fusion_2_models_l1_w_o_bias_83_29.sav'
    # Predict
    load_model_name = 'pytorch_fusion_model_83_29.pt'#'learned_late_fusion_2_models_l2_w_o_bias_83_28.sav'
    #predictor = pickle.load(open(load_model_name, 'rb'))
    predictor = torch.load(load_model_name)
    method = 'learned_fusion'#'learned_fusion'#'sumLogits_softmax'
    result = apply_late_fusion_and_evaluate(cfg, evaluator, val_1, val_2, method, predictor, bayesian=True)
    pdb.set_trace()
    """
    """
    Handle training dataset annotations
    """
    data_set = 'train'
    in_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_RGBT_pairs_3_class.json'
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
    #pdb.set_trace()
    save_file_name = 'train_labels_2_model.npz'#'train_labels_2_model_0219.npz'
    """
    # Get training data    
    X_train, Y_train = get_train_labels(det_1, det_2, anno_train_gt)
    np.savez(save_file_name, X=X_train, Y=Y_train)
    pdb.set_trace()
    """
    """
    # Predict
    load_model_name = 'learned_models/learned_late_fusion_2_models_l1_w_o_bias_83_29.sav'
    predictor = pickle.load(open(load_model_name, 'rb'))
    result = apply_late_fusion_and_evaluate(cfg, evaluator, val_1, val_2, method, predictor)
    """
    print('Loading saved data ...')
    train_data = np.load(save_file_name)
    X_train = train_data['X']
    Y_train = train_data['Y']
    print('Done data preparation.')
    
    from sklearn.linear_model import LogisticRegression
    train_samples = len(Y_train)

    #save_file_name = 'learning_late_fusion_weight_fuse_2_models_l1_w_bias.npz'
    ##########################
    # User Setting
    ##########################
    penalty = 'l1'
    use_bias = False
    method = 'wt_box_fusion' #'avg_box_fusion' #'wt_box_fusion'
    folder = './'#'out/models/'
    random_init = True
    ##########################
    if use_bias:
        save_model_name = folder + 'learned_late_fusion_2_models_'+penalty+'_w_bias.sav'
    else:
        save_model_name = folder + 'learned_late_fusion_2_models_'+penalty+'_w_o_bias.sav'
    
    """
    # Predict
    foler = 'learned_models/'
    load_model_name = foler + 'learned_late_fusion_2_models_l2_w_o_bias_83_28.sav'
    predictor = pickle.load(open(load_model_name, 'rb'))
    result = evaluate(cfg, evaluator, val_1, val_2, predictor, method, bayesian=True)
    pdb.set_trace()
    """
    method = 'learned_fusion'#'sumLogits_softmax'#'learned_fusion'
    max_score = 0
    loop_time = 1
    #class_weight = {0:1.0, 1:3.0, 2:0.8, 3:0.095}
    #class_weight = {0:1.0, 1:3.0, 2:0.8, 3:0.095}
    #class_weight = {0:0.61,1:0.97,2:0.5,3:0.10}
    class_weight = {0:0.61,1:0.97,2:0.6,3:0.10}
    for i in range(loop_time):
        print('------- iteration: ', i, ' -------')
        print('Penalty: ', penalty)
        if use_bias: print('Using bias ...')        
        else: print('Not using bias ...')
        #predictor = LogisticRegression(C=100. / train_samples, penalty=penalty, solver='saga', tol=0.1, max_iter=200, fit_intercept=use_bias)
        predictor = LogisticRegression(C=10, penalty=penalty, solver='saga',verbose=1, tol=0.0001, max_iter=200, fit_intercept=use_bias, class_weight=class_weight)
        predictor.fit(X_train, Y_train)
        #result = evaluate(cfg, evaluator, val_1, val_2, predictor, method)
        #result = apply_late_fusion_and_evaluate(cfg, evaluator, val_1, val_2, method, predictor, bayesian=True)
        result = apply_late_fusion_and_evaluate(cfg, evaluator, val_1, val_2, method, predictor)
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