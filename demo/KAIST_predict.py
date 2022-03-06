import os
import numpy as np
import json
from detectron2.structures import BoxMode
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def get_balloon_dicts(anno_dir,img_dir):
    #img_dir = '/home/jack/MSCV_Capstone/week7/images'
    dataset_dicts = []
    dirs = os.listdir(img_dir)
    idx = 0
    for file in dirs:
        record = {}
        img_name = os.path.join(img_dir, file)
        anno_name = os.path.join(anno_dir,file[:-4]+'.txt')
        #print(img_name)
        #print(anno_name)
        height, width = cv2.imread(img_name).shape[:2]
        record['file_name'] = img_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        #print(record['file_name'])
        fo = open(anno_name,'r')
        lines = fo.readlines()
        objs = []
        for line in lines[1:]:
            line = line.split()
            cx, cy, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            x1 = round(cx)
            y1 = round(cy)
            x2 = round(cx + w, 2)
            y2 = round(cy + h, 2)
            obj = {
                "bbox": [x1, y1,x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)


        record["annotations"] = objs
        record['annotations'] = objs
        dataset_dicts.append(record)
        fo.close()
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog
img_dir = '/home/jack/MSCV_Capstone/fall/week3/new_data/sanitized_images_thermal'
anno_dir = '/home/jack/MSCV_Capstone/fall/week3/new_data/sanitized_annotations'
print('start register---------------------------------------')
for d in ["train",'test']:
    DatasetCatalog.register("kaist_" + d, lambda d=d: get_balloon_dicts(anno_dir+'/' + d,img_dir+'/' + d))
    MetadataCatalog.get("kaist_" + d).set(thing_classes=["person"])
kaist_metadata = MetadataCatalog.get("kaist_train")
print('register done!--------------------------------------')

print('start train---------------------------------')
#------------------train-------------------
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("kaist_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
# cfg.SOLVER.MAX_ITER = 30000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.OUTPUT_DIR = '/home/jack/MSCV_Capstone/fall/week7/final_models/output_thermal_67'
#
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
# print('train done-------------------------')
# print('start test--------------------------')

#-----------------my own test procedure----------------------------

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("kaist_train", )
predictor = DefaultPredictor(cfg)



#---------------------evaluate on testset------------------------------
print('start evaluate on test set-------------------------------------')
save_dir = '/home/jack/MSCV_Capstone/fall/week8/results_low_thre/thermal/test2'
input_dir = '/home/jack/MSCV_Capstone/fall/week3/new_data/sanitized_images_thermal/test'
if not os.path.exists(save_dir): # if it doesn't exist already
    os.makedirs(save_dir)
for filename in os.listdir(r""+input_dir):
    fw = open(save_dir + "/" + filename[:-4]+'.txt','w')
    #print(filename) #just for test
    #img is used to store the image data
    im = cv2.imread(input_dir + "/" + filename)
    outputs1 = predictor(im)
    print(filename)
   # print('color_image',outputs1["instances"].pred_classes)
    #print('color_image',outputs1["instances"].pred_boxes)
    '''
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs1["instances"].to("cpu"))
    cv2.imshow('img1',v.get_image()[:, :, ::-1])
   # cv2.imwrite('fusion_results'+'/'+filename,v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    class_out = outputs1["instances"].pred_classes.cpu().numpy()
    bbox_out = outputs1["instances"].pred_boxes.tensor.cpu().numpy()
    scores_out = outputs1["instances"].scores.cpu().numpy()
    indexes = np.where(class_out==0)
    indexes = indexes[0]
    for index in indexes:
        bbox = bbox_out[index]
        print(bbox)
        score = scores_out[index]
        print(score)
        fw.write('person'+' '+ str(score)+' '+ str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]))
        fw.write("\n")
    fw.close()

#/home/jack/detectron2_repo/detectron2/modeling/meta_arch
#/home/jack/MSCV_Capstone/fall/week4