# FEDER
**Camouflaged Object Detection with Feature Decomposition and Edge Reconstruction**,  CVPR 2023

[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/He_Camouflaged_Object_Detection_With_Feature_Decomposition_and_Edge_Reconstruction_CVPR_2023_paper.pdf)] [[Supplementary material](https://openaccess.thecvf.com/content/CVPR2023/supplemental/He_Camouflaged_Object_Detection_CVPR_2023_supplemental.pdf)] [[Results](https://drive.google.com/drive/folders/1Pho42bHiBhVR0l9KzdOFQgqLzr8mSv9e?usp=sharing)] [[Pretrained models](https://drive.google.com/file/d/1MONpM9auqGlRoyaOKUe6wJgLZ-E6A4Dc/view?usp=sharing)]

#### Authors
[Chunming He](https://chunminghe.github.io/), [Kai Li*](http://kailigo.github.io/), [Yachao Zhang](https://yachao-zhang.github.io/), Longxiang Tang, [Yulun Zhang](https://yulunzhang.com/), [Zhenhua Guo](https://scholar.google.com/citations?user=dbR6bD0AAAAJ&hl=en), [Xiu Li*](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en)

---
> **Abstract:** *Camouflaged object detection (COD) aims to address the tough issue of identifying camouflaged objects visually blended into the surrounding backgrounds. COD is a challenging task due to the intrinsic similarity of camouflaged objects with the background, as well as their ambiguous boundaries. Existing approaches to this problem have developed various techniques to mimic the human visual system. Albeit effective in many cases, these methods still struggle when camouflaged objects are so deceptive to the vision system. In this paper, we propose the FEature Decomposition and Edge Reconstruction (FEDER) model for COD. The FEDER model addresses the intrinsic similarity of foreground and background by decomposing the features into different frequency bands using learnable wavelets. It then focuses on the most informative bands to mine subtle cues that differentiate foreground and background. To achieve this, a frequency attention module and a guidance-based feature aggregation module are developed. To combat the ambiguous boundary problem, we propose to learn an auxiliary edge reconstruction task alongside the COD task. We design an ordinary differential equation-inspired edge reconstruction module that generates exact edges. By learning the auxiliary task in conjunction with the COD task, the FEDER model can generate precise prediction maps with accurate object boundaries. Experiments show that our FEDER model significantly outperforms state-of-the-art methods with cheaper computational and memory costs.*
>
> <p align="center">
> <img width="800" src="Framework.png">
> </p>
---

## Usage



### 1. Prerequisites

> Note that FEDER is only tested on Ubuntu OS with the following environments.

- Creating a virtual environment in terminal: `conda create -n FEDER python=3.8`.
- Installing necessary packages: `pip install -r requirements.txt`

### 2. Downloading Training and Testing Datasets

- Download the [training set](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EQ75AD2A5ClIgqNv6yvstSwBQ1jJNC6DNbk8HISuxPV9QA?e=UhHKSD) (COD10K-train) used for training 
- Download the [testing sets](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EVI0Bjs7k_VIvz4HmSVV9egBo48vjwX7pvx7deXBtooBYg?e=FjGqZZ) (COD10K-test + CAMO-test + CHAMELEON + NC4K ) used for testing

### 3. Training Configuration

- The pretrained model is stored in [Google Drive](https://drive.google.com/file/d/1OmE2vEegPPTB1JZpj2SPA6BQnXqiuD1U/view?usp=share_link). After downloading, please change the file path in the corresponding code.
```bash
python Train.py  --epoch 160  --lr 1e-4  --batchsize 36  --trainsize 36  --train_root YOUR_TRAININGSETPATH  --val_root  YOUR_VALIDATIONSETPATH  --save_path YOUR_CHECKPOINTPATH
```

### 4. Testing Configuration

Our well-trained model is stored in [Google Drive](https://drive.google.com/file/d/1MONpM9auqGlRoyaOKUe6wJgLZ-E6A4Dc/view?usp=sharing). After downloading, please change the file path in the corresponding code.
```bash
python Test.py  --testsize YOUR_IMAGESIZE  --pth_path YOUR_CHECKPOINTPATH  --test_dataset_path  YOUR_TESTINGSETPATH
```

### 5. Evaluation

- Matlab code: One-key evaluation is written in [MATLAB code](https://github.com/DengPingFan/CODToolbox), please follow this the instructions in `main.m` and just run it to generate the evaluation results.

### 6. Results download

The prediction results of our FEDER are stored on [Google Drive](https://drive.google.com/drive/folders/1Pho42bHiBhVR0l9KzdOFQgqLzr8mSv9e?usp=sharing) please check.





## Related Works
[Weakly-Supervised Concealed Object Segmentation with SAM-based Pseudo Labeling and Multi-scale Feature Grouping](https://arxiv.org/abs/2305.11003), arxiv 2023.

[Feature Shrinkage Pyramid for Camouflaged Object Detection with Transformers](https://github.com/ZhouHuang23/FSPNet), CVPR23.

[Concealed Object Detection](https://github.com/GewelsJI/SINet-V2), TPAMI22.

You can see more related papers in [awesome-COD](https://github.com/visionxiang/awesome-camouflaged-object-detection).
## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{He2023Camouflaged,
title={Camouflaged Object Detection with Feature Decomposition and Edge Reconstruction},
author={He, Chunming and Li, Kai and Zhang, Yachao and Tang, Longxiang and Zhang, Yulun and Guo, Zhenhua and Li, Xiu},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023}
}
```

## Concat
If you have any questions, please feel free to contact me via email at chunminghe19990224@gmail.com or hcm21@mails.tsinghua.edu.cn.

## Acknowledgement
The code is built on [SINet V2](https://github.com/GewelsJI/SINet-V2). Please also follow the corresponding licenses. Thanks for the awesome work.
