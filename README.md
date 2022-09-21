# Multimodal Object Detection via Probabilistic Ensembling (Codes updating, unfinished)

[ECCV 2022](https://eccv2022.ecva.net/) Oral presentation

[[project page]](https://mscvprojects.ri.cmu.edu/2020teamc/ "RGBT-detection") 
[[code]](https://github.com/Jamie725/RGBT-detection)
[[video demo]](https://youtu.be/VH7826g8u7c "RGBT-detection")
[[paper]](https://arxiv.org/abs/2104.02904)


**Authors**: Yi-Ting Chen<sup>\*</sup>, 
Jinghao Shi<sup>\*</sup>, 
Zelin Ye<sup>\*</sup>, Christoph Mertz, Deva Ramanan<sup>#</sup>, Shu Kong<sup>#</sup>

![alt text](https://mscvprojects.ri.cmu.edu/2020teamc/wp-content/uploads/sites/33/2020/05/Header.jpg "video demo")






**Abstract** 

Object detection with multimodal inputs can improve many safety-critical systems such as autonomous vehicles (AVs). Motivated by AVs that operate in both day and night, we study multimodal object detection with RGB and thermal cameras, since the latter provides much stronger object signatures under poor illumination. We explore strategies for fusing information from different modalities. Our key contribution is a probabilistic ensembling technique, ProbEn, a simple non-learned method that fuses together detections from multi-modalities. We derive ProbEn from Bayes' rule and first principles that assume conditional independence across modalities. Through probabilistic marginalization, ProbEn elegantly handles missing modalities when detectors do not fire on the same object. Importantly, ProbEn also notably improves multimodal detection even when the conditional independence assumption does not hold, e.g., fusing outputs from other fusion methods (both off-the-shelf and trained in-house). We validate ProbEn on two benchmarks containing both aligned (KAIST) and unaligned (FLIR) multimodal images, showing that ProbEn outperforms prior work by more than 13% in relative performance!



**keywords**
Object Detection, Thermal, infrared camera, RGB-thermal detection, multimodality, multispectral, autonomous driving, sensor fusion, non-maximal suppression, probablistic modeling.



If you find our model/method/dataset useful, please cite our work ([arxiv manuscript](https://arxiv.org/abs/2104.02904)):

    @inproceedings{RGBT-detection,
      title={Multimodal Object Detection via Probabilistic Ensembling},
      author={Chen, Yi-Ting and Shi, Jinghao and Mertz, Christoph and Kong, Shu and Ramanan, Deva},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2022}
    }
