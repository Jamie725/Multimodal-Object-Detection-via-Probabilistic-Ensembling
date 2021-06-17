
## Usage

We provide the training, testing, and visualization code of thermal-only, early-fusion, middle-fusion and Bayesian fusion.

Training:

    python demo_train_thermal_only.py
    python demo_train_early_fusion.py
    python demo_train_middle_fusion.py

Test mAP:

    python demo_mAP_thermal_only.py
    python demo_mAP_early_fusion.py
    python demo_mAP_middle_fusion.py

Bayesian fusion:

    First, you should save predictions from different models 
    -> python save_predictions.py

    Then, you can change and load the predictions in demo_bayesian_fusion.py
    -> python demo_bayesian_fusion.py
