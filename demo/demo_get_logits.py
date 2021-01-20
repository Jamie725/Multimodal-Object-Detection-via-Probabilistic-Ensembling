# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.engine import DefaultTrainer,DefaultPredictor
from detectron2.config import get_cfg,CfgNode
from detectron2.modeling import build_backbone, build_model, build_proposal_generator, build_roi_heads, \
        detector_postprocess
from detectron2.structures import ImageList
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils as utils
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.modeling import build_backbone, build_model, build_proposal_generator, build_roi_heads, \
    detector_postprocess
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.structures import BoxMode, Instances, RotatedBoxes
from detectron2.utils.logger import setup_logger
from detectron2.modeling.box_regression import Box2BoxTransform

import pdb
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os, glob
setup_logger()

class modularFeeding(nn.Module):
    def __init__(self, cfg, predictor, transform_gen, normalizer, device):
        # mean over all
        super(modularFeeding, self).__init__()
        self.predictor = predictor
        self.model = predictor.model
        self.backbone = self.model.backbone
        self.proposal_generator = self.model.proposal_generator
        self.roi_heads = self.model.roi_heads
        self.transform_gen = transform_gen
        self.normalizer = normalizer
        self.box_head = self.roi_heads.box_head
        self.box_predictor = self.roi_heads.box_predictor
        self.device = device

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on = cfg.MODEL.MASK_ON
        self.pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        #pdb.set_trace()
        self.box2box_transform = Box2BoxTransform#self.roi_heads.box2box_transform
        self.smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA#self.roi_heads.smooth_l1_beta

        self.feature_strides = {k: v.stride for k, v in self.backbone.output_shape().items()}
        self.pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)

        self.func_box_pooler = ROIPooler(output_size=self.pooler_resolution, scales=self.pooler_scales,
                                         sampling_ratio=self.sampling_ratio, pooler_type=self.pooler_type, )

    def forward(self, orgImgBGR):
        with torch.no_grad():
            finalInstResult = self.predictor(orgImgBGR)
        #print(finalInstResult)

        orgImgBGR = self.transform_gen.get_transform(orgImgBGR).apply_image(orgImgBGR)
        img2feed = torch.from_numpy(orgImgBGR)
        img2feed = img2feed.permute((2, 0, 1))
        img2feed = img2feed.to(self.device)

        batched_inputs = [{"image": img2feed, "height": img2feed.shape[1], "width": img2feed.shape[-1]}]
        images = [img2feed]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        with torch.no_grad():
            features = self.backbone(images.tensor)

        with torch.no_grad():
            proposals, _ = self.proposal_generator(images, features, None)

        with torch.no_grad():
            results, _ = self.roi_heads(images, features, proposals, None)

            #print(results)

        features_list = [features[f] for f in self.in_features]

        with torch.no_grad():
            # feature 1 1000x256x7x7
            box_features_1 = self.func_box_pooler(features_list, [x.proposal_boxes for x in proposals])

        with torch.no_grad():
            # feature 2  1000x1024
            box_features_2 = self.box_head(box_features_1)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features_2)

        outputs = FastRCNNOutputs(self.box2box_transform, pred_class_logits,
                                  pred_proposal_deltas, proposals, self.smooth_l1_beta, )

        pred_preClsScore = outputs.predict_probs()[0].detach().cpu()

        #pred_instances, filterIDlist = outputs.inference(
        #    self.roi_heads.test_score_thresh, self.roi_heads.test_nms_thresh, self.roi_heads.test_detections_per_img)
        pred_instances, filterIDlist = outputs.inference(
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST, cfg.TEST.DETECTIONS_PER_IMAGE)
           
        #print(pred_instances)

        topCandidateIdx = filterIDlist[0].cpu()

        # topNx81, softmax score
        softmaxScoreList = pred_preClsScore[topCandidateIdx]

        # topNx81, raw logits
        logitsList = pred_class_logits[topCandidateIdx].detach().cpu()
        #print(logitsList)

        # topNx256x7x7
        bboxFeat1 = box_features_1[topCandidateIdx].detach().cpu()

        # topNx1024
        bboxFeat2 = box_features_2[topCandidateIdx].detach().cpu()

        return results, softmaxScoreList, logitsList, bboxFeat1, bboxFeat2


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'good_model/thermal_only/model_0009999.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17

# build model from cfg
model = build_model(cfg)  # returns a torch.nn.Module
model.eval()

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# image preprocessing
image_dir = '/home/jamie/Desktop/Datasets/FLIR/val/thermal_8_bit/'
file_name = 'FLIR_09350.jpeg'
path = os.path.join(image_dir,file_name)
img = cv2.imread(path)
print(img.shape)
print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

backbone = model.backbone
proposal_generator = model.proposal_generator
roi_heads = model.roi_heads
transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(-1, 1, 1)
pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(device).view(-1, 1, 1)
normalizer = lambda x: (x - pixel_mean) / pixel_std

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
# outputs = predictor(img)
# print(outputs)
my_model = modularFeeding(cfg,predictor,transform_gen,normalizer,device)
results, softmaxScoreList, logitsList, bboxFeat1, bboxFeat2 = my_model.forward(img)
print(pred_instances)