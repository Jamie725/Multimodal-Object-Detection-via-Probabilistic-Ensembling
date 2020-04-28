#python test.py
#python demo/demo_draw_pr_curve.py
#python demo/demo_draw.py
#python demo/demo_draw_4_input.py
#python demo/draw_pr.py
#python3 demo/demo_mAP.py # > mAP_orig.log
#python3 demo/demo_mAP_new.py  #> mAP_thermal3_5000_iter.log
#python demo/demo_mAP_6_inputs.py #> mAP_RGBT3_sanity_10000_iter.log
#python demo/demo_mAP_6to4_input.py #> mAP_RGBT3_sanity_10000_iter.log
#python demo/demo_train_6_input.py  #> logs/RGBTTT_lr_0_005_100000_iter_no_dogs_take_4_input_pretrained.log
python demo/demo_train_RGBT_sum.py  > logs/RGBTTT_lr_0_005_70000_iter_no_dogs_take_RGBT_sum_pretrained.log
#python3 demo/demo_train_val.py  > lr_0_0005_5000_iter.log 
#python3 demo/demo_train_loop.py  > logs/lr_0_001_8000_iter_eval_per_1000.log 
#python3 demo/demo_evaluate.py # > train_lr_0_0005_10000_iter_output_val.log 
#python3 demo/demo_train.py #> train_lr_0_0005_10000_iter_output_val.log 
#python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#  --input input1.jpg \
#  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
#python register_dataset.py
