import pdb
import os
import json
import numpy as np
from os.path import isfile, join
import cv2

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


'''
box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
    are the coordinates of the image's top left corner. x1 and y1 are the
    coordinates of the image's bottom right corner.
'''

if __name__ == '__main__':
    data_set = 'train'
    data_folder = 'out/mAP/'
    IOU = 50
    RGB_det_file = data_folder + data_set + '_RGB_predictions_IOU' + str(IOU) + '.json'
    thermal_det_file = data_folder + data_set + '_thermal_predictions_IOU' + str(IOU) + '.json'
    rgb_path = '../../../Datasets/FLIR/' + data_set + '/resized_RGB/'
    thermal_path = '../../../Datasets/FLIR/' + data_set + '/thermal_8_bit/'
    out_folder = 'out/box_comparison/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Read detection results
    RGB_det = json.load(open(RGB_det_file, 'r'))
    thermal_det = json.load(open(thermal_det_file, 'r'))

    num_img = len(thermal_det['image'])
    pixel_thr = 60
    #      person, bike, car
    prior = [0.30, 0.05, 0.65]

    #for i in range(num_img):
    for i in range(0, 20):
        rgb_img_name = RGB_det['image'][i]
        rgb_box = RGB_det['boxes'][i]
        rgb_score = RGB_det['scores'][i]
        rgb_class = RGB_det['classes'][i]

        t_img_name = thermal_det['image'][i].split('.')[0] + '.jpeg'
        t_box = thermal_det['boxes'][i]
        t_score = thermal_det['scores'][i]
        t_class = thermal_det['classes'][i]

        # Get RGB centers
        t_centers = []
        num_t_box = len(t_box)
        for j in range(num_t_box):
            box = t_box[j]
            center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
            t_centers.append(center)

        t_centers = np.array(t_centers)

        # -------- Get thermal centors --------
        num_rgb_box = len(rgb_box)
        t_score_new = t_score.copy()
        
        for j in range(num_rgb_box):
            box = rgb_box[j]
            center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
            center = np.expand_dims(center, axis=0)
            center_rep = np.repeat(center, num_t_box, axis=0)

            dist = abs(t_centers - center_rep)
            dist = np.sqrt(np.sum(dist**2, axis=1))

            dist_min_id = np.argmin(dist)
            candidate_box = []
            candidate_id = []
            t_box_area = []
            # Find out boxes within pixel threshold
            for k in range(len(dist)):
                if dist[k] < pixel_thr:
                    candidate_box.append(t_box[k])
                    candidate_id.append(k)
                    t_box_area.append(get_box_area(t_box[k]))
                    
            rgb_box_area = np.array(get_box_area(box))
            rgb_box_area =  np.expand_dims(rgb_box_area, axis=0)
            rgb_box_area =  np.expand_dims(rgb_box_area, axis=0)
            rgb_box_area_rep = np.repeat(rgb_box_area, len(candidate_box), axis=1)

            dist_area = abs(t_box_area - rgb_box_area_rep)
            if len(dist_area[0]) == 0:
                continue
            else:
                area_min_id = np.argmin(dist_area)
            
            area_ratio = t_box_area/rgb_box_area
            #pdb.set_trace()
            if area_ratio[0][area_min_id] > 1.6:
                continue
            else:
                print('not skipped')

            if dist[dist_min_id] <= pixel_thr and rgb_class[j] == t_class[candidate_id[area_min_id]]:
                out_name = out_folder + rgb_img_name.split('.')[0] + '_' + str(j) + '.jpg'

                #pdb.set_trace()
                visualize_2_frames(rgb_path, rgb_img_name, thermal_path, t_img_name, candidate_id[area_min_id], box, t_box, out_name)
                
                
                # Apply Baysian NMS
                #t_score_new[j] = np.exp(np.log(t_score[j])+np.log(rgb_score[closest_id])-np.log(prior[rgb_class[closest_id]]))
            
        
"""
RGB_image = RGB_det['image'] 
RGB_boxes = RGB_det['boxes']
RGB_scores = RGB_det['scores']
RGB_classes = RGB_det['classes']

thermal_image = thermal_det['image'] 
thermal_boxes = thermal_det['boxes']
thermal_scores = thermal_det['scores']
thermal_classes = thermal_det['classes']
"""
