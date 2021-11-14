import brambox as bb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
#import seaborn as sns
import pdb

def draw_pr_curve(anno_file_name, det_result_name, out_pr_name, thr):
    
    img_name_list = []
    class_label_list = []
    image = []
    class_label = []
    id = []
    x_top_left = []
    y_top_left = []
    width = []
    height = []
    confidence = []

    with open(anno_file_name) as f_anno:
        data = json.load(f_anno)
        num_categories = len(data['categories'])
        num_imgs = len(data['images'])
        
        for i in range(num_categories):
            class_label_list.append(data['categories'][i]['name'])
        
        for i in range(num_imgs):
            img_name_list.append(data['images'][i]['file_name'].split('.')[0])

    with open(det_result_name) as f_result:
        data = json.load(f_result)
        num_detection = len(data)

        for i in range(num_detection):
            
            image.append(img_name_list[data[i]['image_id']])
            class_label.append(class_label_list[data[i]['category_id']])
            id.append(i)
            x_top_left.append(data[i]['bbox'][0])
            y_top_left.append(data[i]['bbox'][1])
            width.append(data[i]['bbox'][2])
            height.append(data[i]['bbox'][3])
            confidence.append(data[i]['score'])
    
    dicts = {'image':image, 'class_label':class_label, 'id':id, 'x_top_left':x_top_left, 'y_top_left':y_top_left, 'width':width, 'height':height, 'confidence':confidence}
    det = pd.DataFrame(dicts)
    anno = bb.io.load('anno_coco', anno_file_name)

    #pdb.set_trace()
    pr = bb.stat.pr(det, anno, thr)  # IoU threshold of 0.5
    ap = bb.stat.ap(pr)

    ax = pr.plot('recall', 'precision', drawstyle='steps', label=f'AP = {round(100*ap, 2)}%')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.savefig(out_pr_name)

if __name__ == '__main__':
    anno_file_name = '../../../Datasets/FLIR/val/thermal_annotations_new.json'
    #anno_file_name = '../../../Datasets/FLIR/train/thermal_annotations_small.json'
    det_result_name = 'output_val/coco_instances_results.json'
    #det_result_name = 'output_val/coco_instances_val_results.json'
    out_pr_name = 'pr.png'
    draw_pr_curve(anno_file_name, det_result_name, out_pr_name, 0.5)