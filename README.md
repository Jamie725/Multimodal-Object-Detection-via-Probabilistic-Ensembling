# Multimodal Object Detection via Probabilistic Ensembling

Update: added score/box fusion options explanation in the run.sh file 

[ECCV 2022](https://eccv2022.ecva.net/) Oral presentation 

[[project page]](https://mscvprojects.ri.cmu.edu/2020teamc/ "RGBT-detection") 
[[code]](https://github.com/Jamie725/RGBT-detection)
[[video demo]](https://youtu.be/VH7826g8u7c "RGBT-detection")
[[paper]](https://arxiv.org/abs/2104.02904)
[[models]](https://drive.google.com/drive/folders/1U1qXYPmts8Xl9xhc1Asb_VpR-_szfNv9?usp=sharing)
[[results]](https://drive.google.com/file/d/1XLjWa2KIrbfjaPGikCjSDIRM9U717Hot/view?usp=sharing)

The results of ProbEn are released! ([KAIST](https://drive.google.com/file/d/1XLjWa2KIrbfjaPGikCjSDIRM9U717Hot/view?usp=sharing) / [FLIR](https://drive.google.com/drive/u/2/folders/1yrvYGEKDwL9lDVdrix8IuRVCGDHibqix))

**Authors**: Yi-Ting Chen<sup>\*</sup>, 
Jinghao Shi<sup>\*</sup>, 
Zelin Ye<sup>\*</sup>, Christoph Mertz, Deva Ramanan<sup>#</sup>, Shu Kong<sup>#</sup>

![alt text](https://mscvprojects.ri.cmu.edu/2020teamc/wp-content/uploads/sites/33/2020/05/Header.jpg "video demo")

For installation, please check INSTALL.md.

## Usage

We provide the training, testing, and visualization code of thermal-only, early-fusion, middle-fusion and Bayesian fusion. Please change the setting for different fusion methods in the code.

Training:

    python demo/FLIR/demo_train_FLIR.py
    
Test mAP:

    python demo/FLIR/demo_mAP_FLIR.py
    
Visualize predicted boxes:
    
    python demo/FLIR/demo_draw_FLIR.py    
    
Probabilistic Ensembling:

First, you should save predictions from different models using demo_FLIR_save_predictions.py

    # Example thermal only
    python demo/FLIR/demo_FLIR_save_predictions.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --fusion_method thermal_only --model_path trained_models/FLIR/models/thermal_only/out_model_thermal_only.pth

    # Example early fusion
    python demo/FLIR/demo_FLIR_save_predictions.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --fusion_method early_fusion --model_path trained_models/FLIR/models/early_fusion/out_model_early_fusion.pth

    # Example middle fusion
    python demo/FLIR/demo_FLIR_save_predictions.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --fusion_method middle_fusion --model_path trained_models/FLIR/models/middle_fusion/out_model_middle_fusion.pth

Then, you can change and load the predictions in demo_probEn.py

    python demo/FLIR/demo_probEn.py --dataset_path /home/jamie/Desktop/Datasets/FLIR/val --prediction_path out/  --score_fusion probEn --box_fusion v-avg

For more example usage, please check run.sh file.


If you find our model/method/dataset useful, please cite our work ([arxiv manuscript](https://arxiv.org/abs/2104.02904)):

    @inproceedings{chen2022multimodal,
      title={Multimodal object detection via probabilistic ensembling},
      author={Chen, Yi-Ting and Shi, Jinghao and Ye, Zelin and Mertz, Christoph and Ramanan, Deva and Kong, Shu},
      booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IX},
      pages={139--158},
      year={2022},
      organization={Springer}
    }
