import pickle
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# coco_eval.eval['precision'][T, R, K, A, M]
# T: IoU region, 0.5:0.95, 0.05 as a step, size 10
# R: Recall, 0:100, size 101
# K: Class ID
# A: Area size, (all, small, medium, large), size 4
# M: maxDets, (1, 10, 100), size 3

def draw_pr(eval_file_name, out_folder):
    with open(eval_file_name, 'rb') as eval_file:
        # Data to be read
        class_id = [0, 1, 2]
        class_name = ['Person', 'Bicycle', 'Car'] #, 'Dog']

        # Load evaluation result
        #coco_eval = json.load(open(eval_file_name, 'r'))
        coco_eval = pickle.load(eval_file)

        x_axis = np.arange(0.0, 1.01, 0.01)
        mAP_all = 0
        # IoU = 0.50:0.95
        for id in range(len(class_id)):
            pr_all = coco_eval.eval['precision'][:,:,class_id[id],0,2]
            mAP_all += np.mean(pr_all)

        mAP_all /= len(class_name)
        #print('Average Precision (AP) @[ IoU=0.50:0.95| area= all | maxDets=100 ] = ', mAP_all)
        
        # IoU = 0.5 / 0.75 results
        print('-----------------------------------------------')
        print('| ', end='')
        for i in range(len(class_name)):
            print(class_name[i], '|', end='')
        print('\n')
        mAP_50 = 0
        mAP_75 = 0
        for id in range(len(class_id)):
            out_fig_name = out_folder + '/PR_' + class_name[id] + '_IoU=0_5.png'
            
            title_name = class_name[id] + ' Precision Recall'
            pr_50 = coco_eval.eval['precision'][0,:,class_id[id],0,2]
            pr_75 = coco_eval.eval['precision'][5,:,class_id[id],0,2]
            mAP_50 += np.mean(pr_50)
            mAP_75 += np.mean(pr_75)
            
            print('| %.4f\t' %(np.mean(pr_50)), end='')

            fig = plt.figure()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.title(title_name)
            plt.grid(True)

            plt.plot(x_axis, pr_50, 'b-', label='IoU=0.5')
            plt.plot(x_axis, pr_75, 'c-', label='IoU=0.75')

            plt.legend(loc='lower left')
            plt.savefig(out_fig_name)
            plt.close(fig)
            #pdb.set_trace()
        mAP_50 /= len(class_name)
        mAP_75 /= len(class_name)
        
        print('\n')
        print('Average Precision (AP) @[ IoU=0.50:0.95| area= all\t| maxDets=100 ] =', mAP_all)
        print('Average Precision (AP) @[ IoU=0.50     | area= all\t| maxDets=100 ] =', mAP_50)
        print('Average Precision (AP) @[ IoU=0.75     | area= all\t| maxDets=100 ] =', mAP_75)

        
        size_id = [1,2,3]
        size_name = ['small', 'medium', 'large']
        for sid in range(len(size_id)):
            mAP_all = 0
            cnt = 0
            for cid in range(len(class_id)):
                pr_all = coco_eval.eval['precision'][:,:,class_id[cid],size_id[sid],2]                
                pr_mean = np.mean(pr_all)
                
                if pr_mean > 0:
                    mAP_all += np.mean(pr_all)
                    cnt += 1

            mAP_all /= cnt
            print('Average Precision (AP) @[ IoU=0.50:0.95| area= {0}\t| maxDets=100 ] = {1}' .format(size_name[sid], mAP_all))

        for cid in range(len(class_id)):
            out_fig_name = out_folder + '/PR_' + class_name[cid] + '_IoU=0_5_small_medium_large_temp.png'
            title_name = class_name[cid] + ' Precision Recall'
        
            pr_small = coco_eval.eval['precision'][0,:,class_id[cid],1,2]
            pr_medium = coco_eval.eval['precision'][0,:,class_id[cid],2,2]
            pr_large = coco_eval.eval['precision'][0,:,class_id[cid],3,2]

            ig = plt.figure()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.title(title_name)
            plt.grid(True)

            plt.plot(x_axis, pr_small, 'b-', label='small')
            plt.plot(x_axis, pr_medium, 'c-', label='medium')
            plt.plot(x_axis, pr_large, 'y-', label='large')

            plt.legend(loc='lower left')
            plt.savefig(out_fig_name)
            plt.close(fig)
            
if __name__ == '__main__':
    out_folder = 'pr_curve/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    #draw_pr('out/mAP/concate_2000_val.out', out_folder)
    draw_pr('out/mAP/FLIR_val_var_box_fusion_gnll.out', out_folder)
    #raw_pr('out/mAP/FLIR_thermal_only_3_class.out', out_folder)