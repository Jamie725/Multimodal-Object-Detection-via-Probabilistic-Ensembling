#####################
# save predictions
#####################
# thermal only
#python demo/FLIR/demo_FLIR_save_predictions.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --fusion_method thermal_only --model_path trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth
# early fusion
#python demo/FLIR/demo_FLIR_save_predictions.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --fusion_method early_fusion --model_path trained_models/FLIR/models/early_fusion/out_model_early_fusion.pth
# middle fusion
#python demo/FLIR/demo_FLIR_save_predictions.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --fusion_method middle_fusion --model_path trained_models/FLIR/models/middle_fusion/out_model_middle_fusion.pth


#####################
# Late fusion
#####################
python demo/FLIR/demo_probEn.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --prediction_path out/ \
        --score_fusion max --box_fusion argmax