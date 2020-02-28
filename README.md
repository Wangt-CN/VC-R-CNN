# Visual Commonsense R-CNN (VC R-CNN)

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.0-%237732a8)

**NEW:** the VC Feature pretrained on MSCOCO is provided. Just have a try!

This repository contains the official PyTorch implementation and the proposed VC feature (beta version) for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "[Visual Commonsense R-CNN](https://arxiv.org/abs/2002.12204)" (The link is the Arxiv version which may be slightly different from our final camera-ready version). For technical details, please refer to:

**Visual Commonsense R-CNN** <br />
[Tan Wang](https://wangt-cn.github.io/), Jianqiang Huang, [Hanwang Zhang](https://www.ntu.edu.sg/home/hanwangzhang/), [Qianru Sun](https://qianrusun1015.github.io) <br />
**CVPR 2020** <br />
**[[Paper](https://arxiv.org/abs/2002.12204)]** <br />

<div align="center">
  <img src="https://github.com/Wangt-CN/Wangt-CN.github.io/blob/master/project/vc-rcnn/framework_github.png" width="600px" />
</div>

### Bibtex
If you find our VC feature and code helpful, please kindly consider citing:

```
@misc{wang2020visual,
    title={Visual Commonsense R-CNN},
    author={Tan Wang and Jianqiang Huang and Hanwang Zhang and Qianru Sun},
    year={2020},
    eprint={2002.12204},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contents
1. Overview
2. Our VC Feature
   - Pretrained on COCO
   - Downstream Models (**To be update**)
3. VC R-CNN Framework
   - **To be update**


## Overview
This projuct aims to build a visual commonsense representation learning framework based on the current [object detection codebase](https://github.com/facebookresearch/maskrcnn-benchmark) with un-/self-supervised learning. The **highlights** of our proposed method and feature are listed below:

1. VC Feature:
   - **Effective**: Our visual commonsense representation encodes the``sense-making'' knowledge between object RoIs with causal intervention rather than just trivial correlation prediction. Compared to the previous widely used [Up-Down Feature](https://github.com/peteanderson80/bottom-up-attention), our VC can be regarded as an effective supplementary knowledge that models the interaction between objects for the downstream tasks. 
   - **Easy to Use**: As we introduced in our paper, the VC Feature is extracted by providing the RoI boxes coordinates. Then the VC Feature can be just concatenated on the previous visual object features (e.g., Up-Down Feature) and ready to roll.  
   - **Easy to Expand**: With a learned VC R-CNN framework, we can easily extract VC Features for any images and prepare them as an ``augmentation feature'' for the currently used representation conveniently.
2. VC R-CNN
   - **Fast, Memory-efficient, Multi-GPU**: Our VC R-CNN framework is based on the well-known [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) from facebook. Therefore, our VC R-CNN just inherit all its advantages. (It's pity that the [detectron2](https://github.com/facebookresearch/detectron2) had not been released when I am working on this project, however maskrcnn-benchmark can be a stable version.)
   - **Support customized dataset**: Users can easily adds COCO-style datasets to train VC R-CNN on other images.



## VC Feature

For easy-to-use, here we directly provide the pretrained VC Features on the entire **MSCOCO dataset** based on the [Up-Down](https://github.com/peteanderson80/bottom-up-attention) feature's boxes in the below links. The features are stored in tar.gz format.



#### 10 - 100 features per image (adaptive):

- COCO 2014 Train/Val Image Features (123K / 6G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1z36lR-CwLjwsJPPxE-phqZTMQCTb5KLV/view?usp=sharing)  &ensp;  [Baidu Drive (key:ec8x)](https://pan.baidu.com/s/1alOZkyGCJSso_znc2i2REA)
- COCO 2014 Testing Image Features (41K / 2G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1PQANKKRdD6j980SjokNTCXV5aQNGW4zS/view?usp=sharing)  &ensp;  [Baidu Drive (key:ec8x)](https://pan.baidu.com/s/1alOZkyGCJSso_znc2i2REA)
- COCO 2015 Testing Image Features (81K / 4G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1U9-EbQI8ZCFe7MvmJXI1xCDTbW2f-E98/view?usp=sharing)  &ensp;  [Baidu Drive (key:ec8x)](https://pan.baidu.com/s/1alOZkyGCJSso_znc2i2REA)

**Ps**: For those who may have no access to the Up-Down feature, here we also provide the feature **after concatenation** and you can directly use without `numpy.concatenate` (The feature dimension is 3072 : 2048+1024):

- [concat] COCO 2014 Train/Val Image Features (123K / 27G) &ensp;   [Google Drive](https://drive.google.com/file/d/1kBnVvph5ISWWljOPeWCdFHWg7lkOp6QX/view?usp=sharing)  
- [concat] COCO 2014 Testing Image Features (41K / 9G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1dSx4BeUJT1NOW6Fdlnmo_3HEbv7zrp1B/view?usp=sharing) 
- [concat] COCO 2015 Testing Image Features (81K / 17G) &ensp;   [Google Drive](https://drive.google.com/file/d/1Sp8w8BTyiVMJjlSUJWvgFFTaH2AAayJQ/view?usp=sharing) 



#### How to use after download

- Unzip the file with command:
```bash
tar -xzvf file_name
```

- The feature format (The shape of each numpy file is [n x 1024]):
```
coco_trainval/test_year
 |---image1.npy
 |---image2.npy
  ...
 |---imageN.npy
```

- Concatenate on the previous feature in the downstream task training.



#### Tips for using in downstream tasks

- We recommend users to **add the dimension** of the start multi-layers (embedding layer, fc and so on) in the downstream networks since the feature size add from 2048 to 3072 (for Up-Down Feature).
- The learning rate can be **slighted reduced**.
- We find the self-attentive operation on feature (e.g., the refining encoder in AoANet) may hurt the effectiveness of our VC Feature. Details can be kindly found at the bottom of Page 7 in our paper.



If you have any questions or concerns, please kindly email to [Tan Wang](wangt97@hotmail.com).
