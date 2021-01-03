import pdb
import os
import json
import numpy as np
from os.path import isfile, join
import cv2
import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances, Boxes
from detectron2.evaluation import FLIREvaluator

def distance(p1, p2):
    dist = sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

def visualize_2_frames(rgb_path, rgb_img_name, thermal_path, t_img_name, closest_id, rgb_box, t_box, out_name):
    rgb_img = cv2.imread(rgb_path + rgb_img_name)
    t_img = cv2.imread(thermal_path + t_img_name)
    
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

def nms(rgb_info, thermal_info):
    # RGB boxes append thermal boxes
    # Order: len(RGB) | len(thermal)
    # Boxes
    boxes = rgb_info['bbox'].copy()
    boxes.extend(thermal_info['bbox'])
    boxes = torch.Tensor(boxes)
    # Scores
    scores = rgb_info['score'].copy()
    scores.extend(thermal_info['score'])
    scores = torch.Tensor(scores)
    # Classes
    classes = rgb_info['class'].copy()
    classes.extend(thermal_info['class'])
    classes = torch.Tensor(classes)
    # Perform nms
    iou_threshold = 0.7
    keep_id = box_ops.batched_nms(boxes, scores, classes, iou_threshold)
    # Add to output
    out_boxes = Boxes(boxes[keep_id])
    out_scores = torch.Tensor(scores[keep_id])
    out_class = torch.Tensor(classes[keep_id])
    return out_boxes, out_scores, out_class

def apply_nms_and_evaluate(evaluator, RGB_det, thermal_det):
    evaluator.reset()
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    num_img = len(thermal_det['image'])
    count_RGB=0
    count_thermal=0
    count_NMS=0

    for i in range(num_img):
        rgb_info = {}
        rgb_info['img_name'] = RGB_det['image'][i]
        rgb_info['bbox'] = RGB_det['boxes'][i]
        rgb_info['score'] = RGB_det['scores'][i]
        rgb_info['class'] = RGB_det['classes'][i]
        
        thermal_info = {}
        thermal_info['img_name'] = thermal_det['image'][i].split('.')[0] + '.jpeg'
        thermal_info['bbox'] = thermal_det['boxes'][i]
        thermal_info['score'] = thermal_det['scores'][i]
        thermal_info['class'] = thermal_det['classes'][i]
        out_boxes, out_scores, out_class = nms(rgb_info, thermal_info)

        count_RGB += len(rgb_info['bbox'])
        count_thermal += len(thermal_info['bbox'])
        count_NMS += len(out_boxes)
       
        file_name = img_folder + rgb_info['img_name'].split('.')[0] + '.jpeg'
        img = cv2.imread(file_name)
        H, W, _ = img.shape
       
        # Handle inputs
        inputs = []
        input_info = {}
        input_info['file_name'] = file_name
        input_info['height'] = H
        input_info['width'] = W
        input_info['image_id'] = thermal_det['image_id'][i]
        input_info['image'] = torch.Tensor(img)
        inputs.append(input_info)

        # Handle outputs
        outputs = []
        out_info = {}
        proposals = Instances([H, W])
        proposals.pred_boxes = out_boxes
        proposals.scores = out_scores
        proposals.pred_classes = out_class
        out_info['instances'] = proposals
        outputs.append(out_info)
        pdb.set_trace()
        evaluator.process(inputs, outputs)
    
    results = evaluator.evaluate()
    if results is None:
        results = {}
    
    avgRGB = count_RGB / num_img
    avgThermal = count_thermal / num_img
    avgNMS = count_NMS / num_img

    print('Avg bbox for RGB:', avgRGB, "average count thermal:", avgThermal, 'average count nms:', avgNMS)
    return results

'''
box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
    are the coordinates of the image's top left corner. x1 and y1 are the
    coordinates of the image's bottom right corner.
'''

if __name__ == '__main__':
    data_set = 'val'
    data_folder = 'out/box_predictions/'
    dataset = 'FLIR'
    IOU = 50
    RGB_det_file = data_folder + data_set + '_RGB_predictions_IOU' + str(IOU) + '.json'
    thermal_det_file = data_folder + data_set + '_thermal_predictions_IOU' + str(IOU) + '.json'
    rgb_path = '../../../Datasets/FLIR/' + data_set + '/resized_RGB/'
    thermal_path = '../../../Datasets/FLIR/' + data_set + '/thermal_8_bit/'
    out_folder = 'out/box_comparison/'
    train_json_path = '../../../Datasets/'+dataset+'/train/thermal_annotations_4_channel_no_dogs.json'
    val_json_path = '../../../Datasets/'+dataset+'/val/RGB_annotations_4_channel_no_dogs.json'
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
    cfg.DATALOADER.NUM_WORKERS = 6
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
    RGB_det = json.load(open(RGB_det_file, 'r'))
    thermal_det = json.load(open(thermal_det_file, 'r'))
    evaluator = FLIREvaluator(dataset, cfg, False, output_dir=out_folder, save_eval=True, out_eval_path='FLIR_noT_val_eval.out')
    result = apply_nms_and_evaluate(evaluator, RGB_det, thermal_det)