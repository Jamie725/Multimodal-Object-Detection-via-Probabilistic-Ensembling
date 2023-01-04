
## Usage

We provide the training, testing, and visualization code of thermal-only, early-fusion, middle-fusion and Bayesian fusion. Please change the setting for different fusion methods in the code.

Training:

    python demo_train_FLIR.py
    
Test mAP:

    python demo_mAP_FLIR.py
    
Visualize predicted boxes:
    
    python demo_draw_FLIR.py    
    
Probabilistic Ensembling:

    First, you should save predictions from different models using demo_FLIR_save_predictions.py
    # Example
    -> python demo/FLIR/demo_FLIR_save_predictions.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --fusion_method thermal_only --model_path trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth

    Then, you can change and load the predictions in demo_probEn.py
    -> python demo/FLIR/demo_probEn.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --prediction_path out/ \
        --score_fusion max --box_fusion argmax

For more example usage, please check run.sh file.