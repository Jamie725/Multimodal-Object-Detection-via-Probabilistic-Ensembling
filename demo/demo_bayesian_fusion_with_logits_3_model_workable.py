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
from detectron2.layers.nms import batched_nms

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
    scores = np.zeros((match_score_vec.shape[0], 4))
    scores[:,:3] = match_score_vec
    scores[:,-1] = 1 - np.sum(match_score_vec, axis=1)
    log_scores = np.log(scores)
    sum_logits = np.sum(log_scores, axis=0)
    exp_logits = np.exp(sum_logits)
    #out_score = exp_logits[pred_class] / np.sum(exp_logits)
    score_norm = exp_logits / np.sum(exp_logits)
    out_score = np.max(score_norm)
    out_class = np.argmax(score_norm)    
    return out_score, out_class
def prob_fusion_multiclass(match_score_vec, pred_class, stds):    
    scores = np.zeros((match_score_vec.shape[0], 4))
    scores[:,:3] = match_score_vec
    scores[:,-1] = 1 - np.sum(match_score_vec, axis=1)    
    invalid = np.where(stds == 0)[0]
    if len(invalid)>0:           
        min_std = sorted(stds)[1]
        stds[invalid] = min_std
        
    sum_div_std = np.sum(1/stds)
    weight = 1/(stds * sum_div_std)
    scores = scores*weight[:,None]
    pdb.set_trace()
    log_scores = np.log(scores)
    sum_logits = np.sum(log_scores, axis=0)
    exp_logits = np.exp(sum_logits)    
    #out_score = exp_logits[pred_class] / np.sum(exp_logits)
    score_norm = exp_logits / np.sum(exp_logits)
    out_score = np.max(score_norm)
    out_class = np.argmax(score_norm)    
    return out_score, out_class

def bayesian_fusion_multiclass_prior(match_score_vec, pred_class, class_prior):
    scores = np.zeros((match_score_vec.shape[0], 4))
    scores[:,:3] = match_score_vec
    scores[:,-1] = 1 - np.sum(match_score_vec, axis=1)
    log_scores = np.log(scores)
    sum_logits = np.sum(log_scores, axis=0) - np.log(class_prior)
    exp_logits = np.exp(sum_logits)
    out_score = exp_logits[pred_class] / np.sum(exp_logits)
    return out_score
    
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
    keep_id = batched_nms(boxes, scores, classes, iou_threshold)

    # Add to output
    out_boxes = boxes[keep_id]
    out_scores = torch.Tensor(scores[keep_id])
    out_class = torch.Tensor(classes[keep_id])
    
    return out_boxes, out_scores, out_class

def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)        
    out_bbox = np.array(bbox) * weight[:,None]
    out_bbox = np.sum(out_bbox, axis=0)    
    return out_bbox

def prob_box_fusion(bbox, score, stds):    
    """
    invalid = np.where(stds == 0)[0]
    if len(invalid) > 0:
        min_std = sorted(stds)[1]
        mean_std = np.mean(stds)
        stds[invalid] = min_std
        #for i in range(len(invalid)):
        #    stds[invalid[i]] = np.mean(stds)
    invalid = np.where(stds == 0)[0]
    """
    """
    valid = np.where(stds > 0)[0]
    if len(valid) < len(stds):
        valid = np.where(stds > 0)[0]
        if len(valid) == 0:
            stds[np.where(stds == 0)] = 1
            #pdb.set_trace()
        else:
            min_std = np.min(stds[np.where(stds > 0)])        
            mean_std = np.mean(stds[np.where(stds > 0)])        
            stds[np.where(stds == 0)] = min_std
        #pdb.set_trace()
    """
    #if len(valid) == 0:
    #    weight_norm = score / np.sum(score)
    #else:
    #import pdb; pdb.set_trace()
    
    #W1 = 1 / [(1/s1 + 1/s2)s1]    
    weight = 1 / (np.sum(1 / stds) * stds)
    weight_norm = weight / np.sum(weight)
    
    #sum_div_stds = np.sum(1/stds)
    #weight = 1/(stds * sum_div_stds)
    #weight_norm = weight / np.sum(weight)
    
    out_bbox = np.array(bbox) * weight_norm[:,None]
    out_bbox = np.sum(out_bbox, axis=0)

    invalid = np.where(stds == 0)[0]
    if len(invalid) > 0:
        pdb.set_trace()
    return out_bbox
def prob_alpha_box_fusion(bbox, score, stds, alpha):    
    stds_alpha = stds * alpha + (1-alpha)
    """
    invalid = np.where(stds_alpha == 0)[0]
    if len(invalid) > 0:
        min_std = sorted(stds_alpha)[1]
        mean_std = np.mean(stds_alpha)
        stds_alpha[invalid] = min_std
        #for i in range(len(invalid)):
        #    stds_alpha[invalid[i]] = np.mean(stds_alpha)
    """
    
    valid = np.where(stds_alpha > 0)[0]
    if len(valid) < len(stds_alpha):
        valid = np.where(stds_alpha > 0)[0]
        if len(valid) == 0:
            stds_alpha[np.where(stds_alpha == 0)] = 1
            #pdb.set_trace()
        else:
            min_std = np.min(stds_alpha[np.where(stds_alpha > 0)])        
            stds_alpha[np.where(stds_alpha == 0)] = min_std
        
    weight = 1 / (np.sum(1 / stds_alpha) * stds_alpha)
    weight_norm = weight / np.sum(weight)

    
    out_bbox = np.array(bbox) * weight_norm[:,None]
    out_bbox = np.sum(out_bbox, axis=0)
    
    """
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight_norm[i] * bbox[i]
    """
    return out_bbox

def prepare_data(info1, info2, info3='', method=None):
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
    if method == 'wt_var_box_fusion':
        vars1 = info1['vars']
        vars2 = info2['vars']
        out_vars = np.concatenate((vars1, vars2), axis=0)
    elif method:
        stds1 = info1['stds']
        stds2 = info2['stds']
        out_stds = np.concatenate((stds1, stds2), axis=0)

    # If more than two detections are fused
    if info3:
        bbox3 = np.array(info3['bbox'])
        score3 = np.array(info3['score'])
        class3 = np.array(info3['class'])
        out_logits['3'] = np.array(info3['class_logits'])
        out_bbox = np.concatenate((out_bbox, bbox3), axis=0)
        out_score = np.concatenate((out_score, score3), axis=0)
        out_class = np.concatenate((out_class, class3), axis=0)

        if method == 'wt_var_box_fusion':            
            vars3 = info3['vars']
            out_vars = np.concatenate((out_vars, vars3), axis=0)
        elif method:
            stds3 = info3['stds']
            out_stds = np.concatenate((out_stds, stds3), axis=0)
    
    if 'prob' in info1.keys():
        prob1 = np.array(info1['prob'])
        prob2 = np.array(info2['prob'])  
        out_prob = np.concatenate((prob1, prob2), axis=0)
        if info3:
            prob3 = np.array(info3['prob'])  
            out_prob = np.concatenate((out_prob, prob3), axis=0)
        
        if method == 'wt_var_box_fusion':
            return out_bbox, out_score, out_class, out_logits, out_prob, out_vars
        elif method:
            return out_bbox, out_score, out_class, out_logits, out_prob, out_stds
        else:    
            return out_bbox, out_score, out_class, out_logits, out_prob
    else:
        if method == 'wt_var_box_fusion':
            return out_bbox, out_score, out_class, out_logits, out_vars
        elif method:
            return out_bbox, out_score, out_class, out_logits, out_stds
        else:
            return out_bbox, out_score, out_class, out_logits

def nms_bayesian(dets, scores, classes, probs, thresh, method, stds=None, var=None):
    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    out_classes = []
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
         
        # If some boxes are matched
        if len(match_score)>0:
            match_score += [original_score]
            match_prob += [original_prob]            
            # Try with different fusion methods
            if method == 'avg_score_bbox':
                final_score = np.mean(np.asarray(match_score))
                match_bbox += [original_bbox]             
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'avg_score':
                final_score = np.mean(np.asarray(match_score))
                final_bbox = original_bbox
            elif method == 'avg_score_wt_score_box':
                final_score = np.mean(np.asarray(match_score))
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            elif method == 'bayesian':
                final_score, out_class = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                out_classes.append(out_class)
                #final_bbox = avg_bbox_fusion(match_bbox)
                final_bbox = original_bbox
            elif method == 'baysian_avg_bbox':
                final_score, out_class = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])                
                #final_score = np.max(match_score)
                out_classes.append(out_class)
                match_bbox += [original_bbox]
                final_bbox = avg_bbox_fusion(match_bbox)
            elif method == 'bayesian_wt_score_box':
                final_score, out_class = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                #final_score = np.max(match_score)
                out_classes.append(out_class)             
                #if final_score < 0.5:
                #    continue            
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            elif method == 'wt_var_box_fusion':
                match_var = list(var[match_ind])
                try:
                    original_var = var[i]
                except:
                    pdb.set_trace()
                match_var += [original_var]
                match_bbox += [original_bbox]
                weights = 1/np.array(match_var)
                final_score, out_class = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                out_classes.append(out_class)
                #final_bbox = original_bbox
                final_bbox = weighted_box_fusion(match_bbox, np.squeeze(weights))
                          
            elif method == 'bayesian_prior_wt_score_box':
                """
                This method is to set different ratios o priors to test how serious priors affect the overall performance (See supplements.)
                """
                p4person = 4
                class_prior = [p4person,1,1,1]
                class_prior /= np.sum(class_prior)
                final_score = bayesian_fusion_multiclass_prior(np.asarray(match_prob), classes[i], class_prior)                
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            elif method == 'prob_fusion':
                match_stds = list(stds[match_ind])
                original_std = stds[i]
                match_stds += [original_std]
                final_score, out_class = prob_fusion_multiclass(np.asarray(match_prob), classes[i], np.array(match_stds))                
                out_classes.append(out_class)
                #final_bbox = avg_bbox_fusion(match_bbox)
                final_bbox = original_bbox
            elif method == 'bayesian_prob_box_fusion':
                final_score, out_class = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                out_classes.append(out_class)                
                #if final_score < 0.5:
                #    continue            
                match_bbox += [original_bbox]
                match_stds = list(stds[match_ind])
                original_std = stds[i]
                match_stds += [original_std]                
                final_bbox = prob_box_fusion(match_bbox, match_score, np.array(match_stds))
            elif method == 'bayesian_prob_alpha_box_fusion':
                final_score, out_class = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                out_classes.append(out_class)
                match_bbox += [original_bbox]
                match_stds = list(stds[match_ind])
                original_std = stds[i]
                match_stds += [original_std]                
                final_bbox = prob_alpha_box_fusion(match_bbox, match_score, np.array(match_stds), 0.4)
            elif method == 'avg_score_prob_box_fusion':
                final_score = np.mean(np.asarray(match_score))
                out_classes.append(classes[i])
                match_bbox += [original_bbox]
                match_stds = list(stds[match_ind])
                original_std = stds[i]
                match_stds += [original_std]
                final_bbox = prob_box_fusion(match_bbox, match_score, np.array(match_stds))
            elif method == 'weighted_box_fusion':
                final_score = np.mean(np.asarray(match_score))
                final_score *= min(3, len(match_score))/3                
                match_bbox += [original_bbox]
                out_classes.append(classes[i])
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            match_scores.append(final_score)
            match_bboxs.append(final_bbox)
        else:
            match_scores.append(original_score)
            match_bboxs.append(original_bbox)
            out_classes.append(classes[i])

        order = order[inds + 1]
        
    assert len(keep)==len(match_scores)
    assert len(keep)==len(match_bboxs)
    assert len(keep)==len(out_classes)

    match_bboxs = match_bboxs
    match_scores = torch.Tensor(match_scores)
    #match_classes = torch.Tensor(classes[keep])
    match_classes = torch.Tensor(out_classes)

    return keep,match_scores,match_bboxs, match_classes

def handle_logits(logits, classes):
    logits1 = logits['1']
    logits2 = logits['2']
    out_logits = np.concatenate((logits1, logits2), axis=0)
    if '3' in logits.keys():
        logits3 = logits['3']
        out_logits = np.concatenate((out_logits, logits3), axis=0)
    return out_logits

def nms_logits(dets, scores, classes, logits, thresh, method):
    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512
    
    out_logits = handle_logits(logits, classes)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    if method == 'logits_fusion':
        pdb.set_trace()
    else:
        order = scores.argsort()[::-1]
    keep = []
    out_class = []
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

        match_score = list(out_logits[match_ind])
        match_bbox = list(dets[match_ind][:,:4])
        original_score = out_logits[i].tolist()        
        original_bbox = dets[i][:4]
        if len(match_score)>0:
            match_score += [original_score]
            if method == 'avgLogits_softmax':
                final_score = np.mean(np.asarray(match_score), axis=0)
                final_score = F.softmax(torch.Tensor(final_score), dim=0)[classes[i]].tolist()
                #final_bbox = avg_bbox_fusion(match_bbox)
                final_bbox = original_bbox
            elif method == 'sumLogits_softmax':
                final_score = np.sum(np.asarray(match_score), axis=0)
                final_score = F.softmax(torch.Tensor(final_score), dim=0)[classes[i]].tolist()
                #final_bbox = avg_bbox_fusion(match_bbox)
                final_bbox = original_bbox
            elif method == 'sumLogits':
                final_score = np.sum(np.asarray(match_score), axis=0)
                class_id = np.argmax(final_score)
                if class_id == 3:
                    pdb.set_trace()
                    continue
                out_class.append(class_id)                
                final_score = np.max(final_score)                
                #final_score = F.softmax(torch.Tensor(final_score), dim=0)[classes[i]].tolist()
                final_bbox = original_bbox
            elif method == 'logits_fusion':
                pdb.set_trace()
    
            match_scores.append(final_score)
            match_bboxs.append(final_bbox)
        else:
            final_score = F.softmax(torch.Tensor(original_score), dim=0)[classes[i]].tolist()
            match_scores.append(final_score)
            match_bboxs.append(original_bbox)
            out_class.append(classes[i])
            
        order = order[inds + 1]
        
    assert len(keep)==len(match_scores)
    assert len(keep)==len(match_bboxs)    
    assert len(keep)==len(out_class)

    match_bboxs = match_bboxs
    match_scores = torch.Tensor(match_scores)
    match_classes = torch.Tensor(out_class)
    #match_classes = torch.Tensor(classes[keep])
    return keep,match_scores,match_bboxs, match_classes

def fusion(method, info_1, info_2, info_3=''):
    if method == 'nms':            
        out_boxes, out_scores, out_class = nms_1(info_1, info_2, info_3=info_3)
    elif method == 'pooling':
        in_boxes, in_scores, in_class, in_logits, in_prob = prepare_data(info_1, info_2, info3=info_3)
        out_boxes = in_boxes
        out_scores = torch.Tensor(in_scores)
        out_class = torch.Tensor(in_class)
    elif method == 'bayesian' or method == 'baysian_avg_bbox' or method == 'avg_score' or method == 'avg_score_bbox' or method == 'avg_score_wt_score_box' or method == 'bayesian_wt_score_box' or method == 'bayesian_prior_wt_score_box' or method == 'weighted_box_fusion':
        threshold = 0.5
        in_boxes, in_scores, in_class, in_logits, in_prob = prepare_data(info_1, info_2, info3=info_3)
        keep, out_scores, out_boxes, out_class = nms_bayesian(in_boxes, in_scores, in_class, in_prob, threshold, method)
    elif method == 'prob_fusion' or method == 'bayesian_prob_box_fusion' or method == 'bayesian_prob_alpha_box_fusion' or method == 'avg_score_prob_box_fusion':
        threshold = 0.5
        in_boxes, in_scores, in_class, in_logits, in_prob, in_stds = prepare_data(info_1, info_2, info3=info_3, method=method)
        keep, out_scores, out_boxes, out_class = nms_bayesian(in_boxes, in_scores, in_class, in_prob, threshold, method, stds=in_stds)
    elif method == 'wt_var_box_fusion':
        threshold = 0.5
        in_boxes, in_scores, in_class, in_logits, in_prob, in_vars = prepare_data(info_1, info_2, info3=info_3, method=method)
        keep, out_scores, out_boxes, out_class = nms_bayesian(in_boxes, in_scores, in_class, in_prob, threshold, method, var=in_vars)
    elif method == 'avgLogits_softmax' or method == 'sumLogits_softmax' or method == 'logits_fusion' or method == 'sumLogits':
        threshold = 0.5
        in_boxes, in_scores, in_class, in_logits, _ = prepare_data(info_1, info_2, info3=info_3)
        keep, out_scores, out_boxes, out_class = nms_logits(in_boxes, in_scores, in_class, in_logits, threshold, method)
    return out_boxes, out_scores, out_class

def draw_box(img, bbox, pred_class, color):
    class_name = ['person', 'bike', 'car']
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 0.8
    # Line thickness of 2 px
    thickness = 2
    color2 = (0, 130, 255)
    for i in range(len(bbox)):
        img = cv2.rectangle(img,  (int(bbox[i][0]+0.5), int(bbox[i][1]+0.5)),  (int(bbox[i][2]+0.5), int(bbox[i][3]+0.5)), color, 2)
    thickness = 2
    for i in range(len(bbox)):
        min_x = max(int(bbox[i][0] - 5), 0)
        min_y = max(int(bbox[i][1] - 5), 0)        
        img = cv2.putText(img, class_name[int(pred_class[i])], (min_x, min_y), font, fontScale, color2, 2, cv2.LINE_AA)
    return img

def apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, method, det_3=''):
    evaluator.reset()
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    print('Method: ', method)

    for i in range(len(det_2['image'])):
        info_1 = {}
        info_1['img_name'] = det_1['image'][i]
        info_1['bbox'] = det_1['boxes'][i]
        info_1['score'] = det_1['scores'][i]
        info_1['class'] = det_1['classes'][i]
        info_1['class_logits'] = det_1['class_logits'][i]
        
        if 'probs' in det_1.keys():
            info_1['prob'] = det_1['probs'][i]
        if 'stds'  in det_1.keys():
            info_1['stds'] = det_1['stds'][i]
        if 'vars' in det_1.keys():
            info_1['vars'] = det_1['vars'][i]
        info_2 = {}
        info_2['img_name'] = det_2['image'][i].split('.')[0] + '.jpeg'
        info_2['bbox'] = det_2['boxes'][i]
        info_2['score'] = det_2['scores'][i]
        info_2['class'] = det_2['classes'][i]
        info_2['class_logits'] = det_2['class_logits'][i]        

        if 'probs' in det_2.keys():
            info_2['prob'] = det_2['probs'][i]
        if 'stds'  in det_2.keys():
            info_2['stds'] = det_2['stds'][i]
        if 'vars' in det_2.keys():
            info_2['vars'] = det_2['vars'][i]
        
        num_detections = int(len(info_1['bbox']) > 0) + int(len(info_2['bbox']) > 0)

        if det_3:
            info_3 = {}
            info_3['img_name'] = det_3['image'][i].split('.')[0] + '.jpeg'
            info_3['bbox'] = det_3['boxes'][i]
            info_3['score'] = det_3['scores'][i]
            info_3['class'] = det_3['classes'][i]
            info_3['class_logits'] = det_3['class_logits'][i]
            if 'probs' in det_3.keys():
                info_3['prob'] = det_3['probs'][i]
            if 'stds' in det_3.keys():
                info_3['stds'] = det_3['stds'][i]
            if 'vars' in det_3.keys():
                info_3['vars'] = det_3['vars'][i]
            num_detections += int(len(info_3['bbox']) > 0)
        
        # No detections
        if num_detections == 0:
            continue
        # Only 1 model detection
        elif num_detections == 1:            
            if len(info_1['bbox']) > 0:
                out_boxes = np.array(info_1['bbox'])
                out_class = torch.Tensor(info_1['class'])
                out_scores = torch.Tensor(info_1['score'])
            elif len(info_2['bbox']) > 0:
                out_boxes = np.array(info_2['bbox'])
                out_class = torch.Tensor(info_2['class'])
                out_scores = torch.Tensor(info_2['score'])
            else:
                if det_3:
                    out_boxes = np.array(info_3['bbox'])
                    out_class = torch.Tensor(info_3['class'])
                    out_scores = torch.Tensor(info_3['score'])
        # Only two models with detections
        elif num_detections == 2:
            if not det_3:
                out_boxes, out_scores, out_class = fusion(method, info_1, info_2)
            else:    
                if len(info_1['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_2, info_3)
                elif len(info_2['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_3)
                else:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_2)
        # All models detected things
        else:
            out_boxes, out_scores, out_class = fusion(method, info_1, info_2, info_3=info_3)
            
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
                        
    results = evaluator.evaluate(out_eval_path='out/mAP/FLIR_bayesian_wt_score_bbox_3_class.out')
    
    if results is None:
        results = {}

    return results


if __name__ == '__main__':
    data_set = 'val'
    data_folder = 'out/box_predictions/3_class/'
    dataset = 'FLIR'
    IOU = 50                 
    time = 'all'
    
    model_1 = 'early_fusion'
    model_2 = 'mid_fusion'
    model_3 = 'thermal_only'
    """
    model_1 = 'BGR_only'
    model_2 = 'thermal_only'
    model_3 = 'mid_fusion'
    """
    if time == 'Day':
        val_file_name = 'thermal_RGBT_pairs_3_class_Day.json'#'thermal_annotations_4_channel_no_dogs_Day.json'
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_with_multiclass_prob_score_Day.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_with_multiclass_prob_score_Day.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_with_multiclass_prob_score_Day.json'
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_Day.json'
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_Day.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_Day.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_Day.json'
        """
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90_Day.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90_Day.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90_Day.json'
        """
    elif time == 'Night':
        val_file_name = 'thermal_RGBT_pairs_3_class_Night.json'#'thermal_annotations_4_channel_no_dogs_Night.json'
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_with_multiclass_prob_score_Night.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_with_multiclass_prob_score_Night.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_with_multiclass_prob_score_Night.json'
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_Night.json'
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_Night.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_Night.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_Night.json'
        """
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90_Night.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90_Night.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90_Night.json'
        """
    else:
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_with_multiclass_prob_score.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_with_multiclass_prob_score.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_with_multiclass_prob_score.json'
        """
        #"""
        # Most commonly used
        #det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        #det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        #det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_gnll.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_gnll.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_gnll.json'
        
        #"""
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_1000_proposals.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_1000_proposals.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_1000_proposals.json'
        """
        
        #det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90.json'
        #det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90.json'
        #det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr_IoU_90.json'
        
        #"""
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_w_score_higher_than_thr.json'
        """
        """
        det_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_dropout_1.json'
        det_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_dropout_1.json'
        det_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score_w_uncertainty_from_proposals_dropout_1.json'
        """
        #"""
        """
        det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_3_class_with_multiclass_prob_score_thr_30.json'
        det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_3_class_with_multiclass_prob_score_thr_30.json'
        det_file_3 = data_folder + 'val_mid_fusion_predictions_IOU50_3_class_with_multiclass_prob_score_thr_30.json'
        """
        val_file_name = 'thermal_RGBT_pairs_3_class.json'#'thermal_annotations_4_channel_no_dogs_3_class.json'#'thermal_RGBT_pairs_3_class.json'
    
    print('detection file 1:', det_file_1)
    print('detection file 2:', det_file_2)
    print('detection file 3:', det_file_3)
    
    path_1 = '../../../Datasets/FLIR/' + data_set + '/resized_RGB/'
    path_2 = '../../../Datasets/FLIR/' + data_set + '/thermal_8_bit/'
    out_folder = 'out/box_comparison/'
    
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
    cfg.OUTPUT_DIR = out_folder
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = (dataset, )
    cfg.INPUT.FORMAT = 'BGR'
    cfg.INPUT.NUM_IN_CHANNELS = 3
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    
    # Read detection results
    det_1 = json.load(open(det_file_1, 'r'))
    det_2 = json.load(open(det_file_2, 'r'))
    det_3 = json.load(open(det_file_3, 'r'))
    
    evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='out/mAP/FLIR_Baysian_'+data_set+'_wt_box_fusion.out') 
    """
    Method lists: 'bayesian_prior_wt_score_box': This is for tuning different background prior
                  'bayesian_wt_score_box'                  
                  'sumLogits'
                  'sumLogits_softmax'
                  'avgLogits_softmax'
                  'baysian_avg_bbox'
                  'avg_score'
                  'avg_score_wt_score_box'
                  'avg_score_bbox': same as top-k voting
                  'pooling'
                  'bayesian'
                  'logits_fusion'
                  'nms'
                  'prob_fusion'
                  'bayesian_prob_box_fusion'
                  'bayesian_prob_alpha_box_fusion'
                  'avg_score_prob_box_fusion'
                  'weighted_box_fusion'
                  'wt_var_box_fusion'
    """    
    method = 'nms'#'logits_fusion'
    # PID: 18492
    # 3 inputs
    result = apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, method, det_3=det_3)
    # 2 inputs only
    #result = apply_late_fusion_and_evaluate(cfg, evaluator, det_1, det_2, method)