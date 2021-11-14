from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import torch
import cv2
import pdb

class FLIRDataset(Dataset):
    def __init__(self, X_train='', Y_train='', img_folder='', det_1='', det_2='', anno=''):
        super().__init__()
        if det_1:        
            self.det_1 = det_1
            self.det_2 = det_2
            self.anno = anno
            self.img_folder = img_folder
            X_train, Y_train = self.train_late_fusion(det_1, det_2, anno)
        self.X = X_train
        self.Y = Y_train
        self.num_classes = 4
    
    def __len__(self):
        return len(self.X)
    
    def train_late_fusion(self, det_1, det_2, anno):
        img_folder = self.img_folder
        num_img = len(det_2['image'])
        count_1=0
        count_2=0
        #method = 'train'
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
                    score_results, class_results, out_bboxs = nms_multiple_box(in_boxes, in_scores, in_class, in_logits, 0.5, num_det, method)
                elif len(info_2['bbox']) == 0:
                    #print('model 2 missing detection')
                    in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt_1_det(info_1, info_gt=info_gt)
                    score_results, class_results, out_bboxs = nms_multiple_box(in_boxes, in_scores, in_class, in_logits, 0.5, num_det, method)
                else:
                    in_boxes, in_scores, in_class, in_logits, in_prob, num_det = prepare_data_gt(info_1, info_2, info_gt=info_gt)
                    score_results, class_results, out_bboxs = nms_multiple_box(in_boxes, in_scores, in_class, in_logits, 0.5, num_det, method)

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
        print('# of instances: ', cnt_inst)
        return X, Y

    
    def __getitem__(self, index):
        X_train = self.X[index]
        X_train = torch.Tensor(X_train)
        
        """
        Y_train = np.zeros(self.num_classes)
        Y_train[int(self.Y[index])] = 1.0
        Y_train = torch.Tensor(Y_train)
        """
        Y_train = torch.as_tensor(self.Y[index]).type(torch.LongTensor)
        
        #Y_train = torch.Tensor(Y_train)
        #pdb.set_trace()
        return X_train, Y_train

class learnFusionModel(nn.Module):
    def __init__(self, use_bias, random_init):
        super().__init__()
        if random_init:
            self.weights = nn.Parameter(torch.rand((1,8)))
        else:
            self.weights = nn.Parameter(torch.ones((1,8)))
        
        if use_bias:
            #self.bias = nn.Parameter(torch.rand(1,4))
            self.bias = nn.Parameter(torch.zeros(1,4))
        
        self.use_bias=use_bias
        
            
    def forward(self, X):
        weighted_X = torch.mul(X, self.weights)
        if self.use_bias:    
            out = weighted_X[:,:4] + weighted_X[:,4:] + self.bias
        else:
            out = weighted_X[:,:4] + weighted_X[:,4:]
        return out

