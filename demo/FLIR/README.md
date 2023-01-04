
## Usage

We provide the training, testing, and visualization code of thermal-only, early-fusion, middle-fusion and Bayesian fusion. Please change the setting for different fusion methods in the code.

Training:

    python demo_train_FLIR.py
    
Test mAP:

    python demo_mAP_FLIR.py
    
Visualize predicted boxes:
    
    python demo_draw_FLIR.py    
    
Probabilistic Ensembling(updating, not finished, will release them soon):
    (The current files here are the previous verison. Will update to latest version soon!)
    
    First, you should save predictions from different models 
    -> python save_predictions.py

    Then, you can change and load the predictions in demo_bayesian_fusion.py
    -> python demo_bayesian_fusion.py
