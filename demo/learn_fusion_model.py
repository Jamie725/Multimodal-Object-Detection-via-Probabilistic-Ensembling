import torch
import torch.nn.functional as F
import cv2
import torch.nn as nn
import numpy as np
import json
import pdb
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
from FLIR_dtaset import FLIRDataset, learnFusionModel
from torch.utils.data import DataLoader

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

def avg_bbox_fusion(match_bbox_vec):
    avg_bboxs = np.sum(match_bbox_vec,axis=0) / len(match_bbox_vec)
    return avg_bboxs

def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight[i] * bbox[i]
    return out_bbox

def nms_multiple_box_eval(dets, scores, classes, logits, thresh, num_det, method):
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
                temp_score[0, model_id_list[k]*4:(model_id_list[k]+1)*4] += logits[match_ind[k]]
             
            match_logits = np.concatenate((match_logits, temp_score))
            match_bbox += [original_bbox]
            
            if method == 'wt_box_fusion':
                match_score.append(original_score)
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            elif method == 'avg_box_fusion':
                final_bbox = avg_bbox_fusion(match_bbox)
            else:
                final_bbox = original_bbox
            match_bboxs.append(final_bbox)
            # Matched groundtruth ID
            match_id.append(match_ind)
           
            #pdb.set_trace()
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
        order = order[inds + 1] 
    
    match_logits = match_logits[1:]

    assert len(match_bboxs)==len(match_logits)
  
    match_bboxs = match_bboxs
    match_logits = torch.Tensor(match_logits)
  
    return match_logits, match_bboxs

def evaluate(cfg, evaluator, det_1, det_2, anno, predictor, method, bayesian=False):
    evaluator.reset()
    predictor.eval()
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
            #print('model 1 miss detected')
            in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_1_det(info_2)
        elif len(info_2['bbox']) == 0:
            #print('model 2 miss detected')
            in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_1_det(info_1)
        else:
            in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt(info_1, info_2)
        score_results, box_results = nms_multiple_box_eval(in_boxes, in_scores, in_class, in_logits, 0.5, num_det, method)
        
        if bayesian:
            # summing logits
            sum_logits = score_results[:,:4] + score_results[:,4:]
            pred_prob_multiclass = F.softmax(torch.Tensor(sum_logits)).tolist()
            out_scores = np.max(pred_prob_multiclass, axis=1)
            out_class = np.argmax(pred_prob_multiclass, axis=1)
        else:
            scores = predictor(torch.Tensor(score_results).cuda(0))
            pred_prob_multiclass = F.softmax(scores)
            out_scores = np.max(pred_prob_multiclass.tolist(), axis=1)
            out_class = np.argmax(pred_prob_multiclass.tolist(), axis=1)
        #score_results /= np.linalg.norm(score_results+0.001, axis=1,keepdims=True)
        #score_results[:,:4] /= np.linalg.norm(score_results[:,:4]+0.001, axis=1,keepdims=True)
        #score_results[:,4:] /= np.linalg.norm(score_results[:,4:]+0.001, axis=1,keepdims=True)
        #pdb.set_trace()
        

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
            elif method == 'sumLogits':
                final_score = np.sum(np.asarray(match_score), axis=0)
                final_score = np.max(final_score)
                final_bbox = avg_bbox_fusion(match_bbox)
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
                final_bbox = avg_bbox_fusion(match_bbox)
                """
                # Weighted bboxs
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_prob)
                #"""
                #pdb.set_trace()
                #"""
                for k in range(len(match_ind)):            
                    temp_score[0, model_id_list[k]*4:(model_id_list[k]+1)*4] += temp_logits[match_ind[k]]
                #"""
                """
                # Fetch only the highest score when 3 bbox matched
                if len(match_ind) > 2:
                    id_0 = np.where(model_id_list == 0)
                    id_1 = np.where(model_id_list == 1)
                    if len(id_0[0]) > 1:
                        keep_id = np.argmax(match_prob[id_0[0]])
                        #pdb.set_trace()
                        temp_score[0, 4:] += temp_logits[match_ind[id_1[0][0]]]
                        temp_score[0, :4] += temp_logits[match_ind[id_0[0][keep_id]]]
                    else:
                        keep_id = np.argmax(match_prob[id_1[0]])
                        #pdb.set_trace()
                        temp_score[0, 4:] += temp_logits[match_ind[id_1[0][keep_id]]]
                        temp_score[0, :4] += temp_logits[match_ind[id_0[0][0]]]
                else: 
                    for k in range(len(match_ind)):
                        temp_score[0, model_id_list[k]*4:(model_id_list[k]+1)*4] += temp_logits[match_ind[k]]
                """        
                final_logits = np.concatenate((final_logits, temp_score))
                sum_score = temp_score[0,:4] + temp_score[0,4:]
                sum_score_softmax = F.softmax(torch.Tensor(sum_score), dim=0)[classes[i]].tolist()
                logits_softmax.append(sum_score_softmax)
                match_case.append(0)
            elif method == 'logRegression':
                # SumLogits
                final_score = np.sum(np.asarray(match_score), axis=0)
                final_score = F.softmax(torch.Tensor(final_score), dim=0)[classes[i]].tolist()
                # Prepare out logits list
                model_id_list = determine_model(num_det, match_ind)
                temp_score = np.zeros((1,8))
                temp_logits = np.zeros((num_det[0] + num_det[1], 4))
                temp_logits[:num_det[0],:] = logits['1']
                temp_logits[num_det[0]:,:] = logits['2']
                final_bbox = avg_bbox_fusion(match_bbox)
                """
                # Weighted bboxs
                match_bbox += [original_bbox]
                final_bbox = weighted_box_fusion(match_bbox, match_prob)
                #"""
                #pdb.set_trace()
                #"""
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
            if method == 'learned_fusion' or method == 'logRegression':
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
    if method == 'learned_fusion' or method == 'logRegression':
        final_logits = final_logits[1:, :]
        assert len(keep) == len(final_logits)
        return keep,match_scores,match_bboxs, match_classes, final_logits, logits_softmax, match_case
    else:
        return keep,match_scores,match_bboxs, match_classes

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
def handle_logits(logits, classes):
    logits1 = logits['1']
    logits2 = logits['2']
    out_logits = np.concatenate((logits1, logits2), axis=0)
    if '3' in logits.keys():
        logits3 = logits['3']
        out_logits = np.concatenate((out_logits, logits3), axis=0)
    return out_logits
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
    elif method == 'sumLogits':
        in_boxes, in_scores, in_class, in_logits, _, num_det = prepare_data(info_1, info_2, info3=info_3)
        keep, out_scores, out_boxes, out_class = nms_logits(in_boxes, in_scores, in_class, in_logits, threshold, method)
    elif method == 'learned_fusion' or method == 'logRegression':
        in_boxes, in_scores, in_class, in_logits, _, num_det = prepare_data(info_1, info_2, info3=info_3)
        keep, out_scores, out_boxes, out_class, out_logits, logits_score, match_case = nms_logits(in_boxes, in_scores, in_class, in_logits, threshold, method, num_det=num_det, predictor=predictor)
        return out_boxes, out_scores, out_class, out_logits, logits_score, match_case
    elif method == 'train':
        in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt(info_1, info_2, info_gt=info_gt)
        keep, out_scores, out_boxes, out_class, out_logits = nms_logits(in_boxes, in_scores, in_class, in_logits, threshold, method, num_det=num_det)
    else:
        print('No matched method:', mehtod,' found!')
    return out_boxes, out_scores, out_class

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
                if method == 'learned_fusion' or method == 'logRegression':
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
            pred_logits = predictor(torch.Tensor(out_logits).cuda(0))
            pred_prob_multiclass = F.softmax(pred_logits, dim=1).tolist()
            out_scores = np.max(pred_prob_multiclass, axis=1)
            out_class = np.argmax(pred_prob_multiclass, axis=1)
        elif method == 'logRegression':
            pred_prob_multiclass = predictor.predict_proba(out_logits)
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
    
    print('Avg bbox for RGB:', avgRGB, "average count thermal:", avgThermal, 'average count nms:', avgNMS)
    return results
def get_cfg_function(out_folder):
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
    return cfg

if __name__ == '__main__':
    ###### User settings ######
    time = 'all'
    data_folder = 'out/box_predictions/3_class/'
    data_set = 'val'
    save_file_name = 'train_labels_2_model.npz'
    to_shuffle = True
    use_bias = True
    max_epoch = 30
    batch_size = 1024*64
    gamma = 0.9
    lr = 1e-3
    method = 'wt_box_fusion'
    out_folder = 'out/box_comparison/'
    dataset = 'FLIR'
    weight_decay = 0.0001
    momentum = 0.9
    random_init = False
    ###########################
    model_1 = 'early_fusion'
    model_2 = 'mid_fusion'
    model_3 = 'thermal_only'
    
    if time == 'Day':
        val_file_name = 'thermal_annotations_4_channel_no_dogs_Day.json'#'RGB_annotations_4_channel_no_dogs.json'#'thermal_annotations_4_channel_no_dogs_Day.json'#
        det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_'+time+'_with_logits.json'
        det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_'+time+'_with_logits.json'
    elif time == 'Night':
        val_file_name = 'thermal_annotations_4_channel_no_dogs_Night.json'
        det_file_1 = data_folder + 'val_thermal_only_predictions_IOU50_'+time+'_with_logits.json'
        det_file_2 = data_folder + 'val_early_fusion_predictions_IOU50_'+time+'_with_logits.json'
    else:
        """
        3 class with multiclass probability score
        """
        val_file_name = 'thermal_annotations_4_channel_no_dogs.json'
        # Training data
        det_file_1 = data_folder + 'train_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_2 = data_folder + 'train_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        det_file_3 = data_folder + 'train_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        
        # Validation data
        val_file_1 = data_folder + 'val_'+model_1+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        val_file_2 = data_folder + 'val_'+model_2+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
        val_file_3 = data_folder + 'val_'+model_3+'_predictions_IOU50_3_class_with_multiclass_prob_score.json'
    
    path_1 = '../../../Datasets/FLIR/' + data_set + '/resized_RGB/'
    path_2 = '../../../Datasets/FLIR/' + data_set + '/thermal_8_bit/'
    
    # Read validation data
    val_1 = json.load(open(val_file_1, 'r'))
    val_2 = json.load(open(val_file_2, 'r'))
    # Load training data
    print('Loading saved data ...')
    train_data = np.load(save_file_name)
    X_train = train_data['X']
    Y_train = train_data['Y']
    print('Done data preparation.')
    
    """
    Handle validation dataset annotations
    """
    data_set = 'val'
    in_anno_file = '../../../Datasets/FLIR/'+data_set+'/thermal_annotations_4_channel_no_dogs.json'
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

    print('Finish getting groundtruth validation data.')
    #########################################################
    val_json_path = '../../../Datasets/'+dataset+'/val/' + val_file_name
    val_folder = '../../../Datasets/FLIR/val/thermal_8_bit'
    # Register dataset
    dataset = 'FLIR_val'
    register_coco_instances(dataset, {}, val_json_path, val_folder)
    FLIR_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = DatasetCatalog.get(dataset)
    
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    dataset_train = FLIRDataset(X_train=X_train, Y_train=Y_train, img_folder=img_folder)
    batch_size = len(X_train)
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=to_shuffle,
        num_workers=16,
    )
    cfg = get_cfg_function(out_folder)
    evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='out/mAP/FLIR_pytorch_learning_fusion.out')
    
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    model = learnFusionModel(use_bias=use_bias, random_init=random_init)
    model = model.train()
    model = model.to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    #pdb.set_trace()
    best_mAP = 0
    best_epoch = 0
    out_model_name = 'pytorch_fusion_model.pt'
    """
    # Load and predict
    out_model_name = 'pytorch_fusion_model.pt'
    model = torch.load(out_model_name)
    pdb.set_trace()
    method = 'sumLogits'#'learned_fusion'
    result = apply_late_fusion_and_evaluate(cfg, evaluator, val_1, val_2, method, model)
    #result = evaluate(cfg, evaluator, val_1, val_2, anno_val_gt, model, method, bayesian=False)
    pdb.set_trace()
    """
    """
    loss_list = []
    AP_list = []
    method = 'learned_fusion'
    class_weight = torch.Tensor([1,2,1,1]).cuda(0)
    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        for idx, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            X, Y = X.to(device), Y.to(device)
            pred_X = model(X)
            criterion = nn.CrossEntropyLoss(weight=class_weight) #(array([23234.,  3937., 41005., 11777.]), array([0.  , 0.75, 1.5 , 2.25, 3.  ]), <BarContainer object of 4 artists>)
            loss = criterion(pred_X, Y)
            total_loss += loss
            loss_list.append(loss.item())
                      
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, ' loss:',total_loss)
        #result = evaluate(cfg, evaluator, val_1, val_2, anno_val_gt, model, method, bayesian=True)
        result = apply_late_fusion_and_evaluate(cfg, evaluator, val_1, val_2, method, model)
        AP_list.append(result['bbox']['AP50'])
        if result['bbox']['AP50'] > best_mAP:
            best_mAP = result['bbox']['AP50']
            best_epoch = epoch
            torch.save(model, out_model_name)
        print('Highest mAP:', best_mAP, ' best epoch:', best_epoch)
        print('----------------------------------')
    """
    ###################################################
    # sklearn Logistic Regression
    ###################################################
    from sklearn.linear_model import LogisticRegression
    train_samples = len(Y_train)

    ##########################
    # User Setting
    ##########################
    penalty = 'l1'
    use_bias = False
    method = 'logRegression' #'avg_box_fusion' #'wt_box_fusion'
    folder = 'out/models/'
    class_weight = {0:1.4, 1:2.4, 2:1.0, 3:0.3}
    #class_weight = {0:0.7, 1:1.2, 2:0.5, 3:0.15}
    ##########################
    if use_bias:
        save_model_name = 'learned_late_fusion_2_models_'+penalty+'_w_bias.sav'
    else:
        save_model_name = 'learned_late_fusion_2_models_'+penalty+'_w_o_bias.sav'

    if use_bias: print('Using bias ...')        
    else: print('Not using bias ...')
    #predictor = LogisticRegression(C=50. / train_samples, penalty=penalty, solver='saga', tol=0.1, max_iter=200, fit_intercept=use_bias)
    predictor = LogisticRegression(C=50, penalty=penalty, solver='saga',verbose=1, tol=0.00001, max_iter=200, fit_intercept=use_bias, class_weight=class_weight)
    predictor.fit(X_train, Y_train)            
    result = apply_late_fusion_and_evaluate(cfg, evaluator, val_1, val_2, method, predictor)
    #pdb.set_trace()
    #result = evaluate(cfg, evaluator, val_1, val_2, anno_val_gt, predictor, method)
