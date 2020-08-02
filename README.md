# Visual Commonsense R-CNN (VC R-CNN)

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.0-%237732a8)


**[NEW]:** We have provided the training code of VC R-CNN and detailed readme file. :star2:

**[NEW]:** the VC Feature pretrained on MSCOCO is provided. Just have a try! :star2:

This repository contains the official PyTorch implementation and the proposed VC feature for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "[Visual Commonsense R-CNN](https://arxiv.org/abs/2002.12204)". For technical details, please refer to:

**Visual Commonsense R-CNN** <br />
[Tan Wang](https://wangt-cn.github.io/), Jianqiang Huang, [Hanwang Zhang](https://www.ntu.edu.sg/home/hanwangzhang/), [Qianru Sun](https://qianrusun1015.github.io) <br />
**IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020** <br />
**Key Words: &nbsp;Causal Intervention; &nbsp;Visual Common Sense; &nbsp;Representation Learning** <br />
**[[Paper](https://arxiv.org/abs/2002.12204)]**, **[[Zhihu Article](https://zhuanlan.zhihu.com/p/111306353)]**, **[[15min Slides](https://github.com/Wangt-CN/Wangt-CN.github.io/blob/master/project/vc-rcnn/slides_15min_vc_rcnn.pdf)]**, **[[Video](https://www.youtube.com/watch?v=iL6m2mVVzpo&list=PL4DwY1suLMkdjiU9lKph6wQbTbKeGP0e_&index=7)]** <br />

<div align="center">
  <img src="https://github.com/Wangt-CN/Wangt-CN.github.io/blob/master/project/vc-rcnn/framework_github.png" width="600px" />
</div>

### Bibtex
If you find our VC feature and code helpful, please kindly consider citing:

```
@inproceedings{wang2020visual,
  title={Visual commonsense r-cnn},
  author={Wang, Tan and Huang, Jianqiang and Zhang, Hanwang and Sun, Qianru},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10760--10770},
  year={2020}
}
```

## Contents
1. [Overview](overview)
   - [Highlights](#highlights)
   - [What can you get from this repo? [The Road Map]](#what-can-you-get-from-this-repo-the-road-map)
2. [VC Feature](#vc-feature)
   - [Pretrained on COCO](#10-100-vc-features-per-image)
   - [Downstream Vision & Language Tasks](#downstream-vision--language-tasks)
3. [VC R-CNN Framework](#vc-r-cnn-framework)
   - [Installation](#installation)
   - [Perform Training on COCO Dataset](#perform-training-on-coco-dataset)
   - [Evaluation (Feature Extraction)](#evaluation-feature-extraction)
   - [Add your Customized Dataset](#add-your-customized-dataset)
4. [Our experience you may need](#our-experience-you-may-need)

   


## Overview
This project aims to build a visual commonsense representation learning framework based on the current [object detection codebase](https://github.com/facebookresearch/maskrcnn-benchmark) with un-/self-supervised learning. 

### Highlights:

1. VC Feature
   - **Effective**: Our visual commonsense representation encodes the``sense-making'' knowledge between object RoIs with causal intervention rather than just trivial correlation prediction. Compared to the previous widely used [Up-Down Feature](https://github.com/peteanderson80/bottom-up-attention), our VC can be regarded as an effective supplementary knowledge that models the interaction between objects for the downstream tasks. 
   - **Easy to Use**: As we introduced in our paper, the VC Feature is extracted by providing the RoI boxes coordinates. Then the VC Feature can be **just concatenated** on the previous visual object features (e.g., Up-Down Feature) and ready to roll.  (**Ps**: the concatenation maybe too simple for some cases or tasks, users can try something else and welcome feedback.)
   - **Easy to Expand**: With a learned VC R-CNN framework, we can easily extract VC Features for any images and prepare them as an ``augmentation feature'' for the currently used representation conveniently.
2. VC R-CNN
   - **Fast, Memory-efficient, Multi-GPU**: Our VC R-CNN framework is based on the well-known [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) from facebook. Therefore, our VC R-CNN just inherit all its advantages. (It's pity that the [detectron2](https://github.com/facebookresearch/detectron2) had not been released when I am working on this project, however maskrcnn-benchmark can be a stable version.)
   - **Support customized dataset**: Users can easily add COCO-style datasets to train VC R-CNN on other images.



### What can you get from this repo? [The Road Map]

- **I want to use your VC Feature pretrained on COCO:** 
  - Download the [VC Feature on COCO](#vc-feature) and concatenate it on Up-Down feature for usage.
  - You can also try other methods to use the VC Feature rather than just concatenation.
- **I want to retrain your VC R-CNN on COCO:**
  - [Perform Training on COCO Dataset](#perform-training-on-coco-dataset)
- **I want to train the VC R-CNN on my own dataset and extract VC Features:**
  - [Add your Customized Dataset](#add-your-customized-dataset)
  - [Perform Training on COCO Dataset](#perform-training-on-coco-dataset)
  - [Evaluation (Feature Extraction)](#evaluation-feature-extraction)

</br>

## VC Feature

For easy-to-use, here we directly provide the pretrained VC Features on the entire **MSCOCO dataset** based on the [Up-Down](https://github.com/peteanderson80/bottom-up-attention) feature's boxes in the below links (The link is updated). The features are stored in tar.gz format. The previous features can be found in [OLD_FEATURE](OLD_FEATURE.md).



#### 10-100 VC Features per image:

- COCO 2014 Train/Val Image Features (123K / 5G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1O-JAYhdF3z8fkLivXZzllT8PotV1MlRv/view?usp=sharing)  &ensp; 
- COCO 2014 Testing Image Features (41K / 2G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1B83Av5H9RtR4-7vw5U2yQun1Ws9rS76Y/view?usp=sharing)  &ensp; 
- COCO 2015 Testing Image Features (81K / 3G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1b42gwXcT7lmyApzNJzcWXC7IAnlMUPnT/view?usp=sharing)  &ensp; 



#### 10-100 Updown Features per image:

For those who may have no access to the Up-Down feature, here we also provide the **Updown feature** here. Then you can directly use `numpy.concatenate` (The feature dimension is 3072 : 2048+1024):

- COCO 2014 Train/Val Image Features (123K / 21G) &ensp;   [Google Drive](https://drive.google.com/file/d/1J62N8HLjNaPell0UdByMyt-bbl8UGlSL/view?usp=sharing)  
- COCO 2014 Testing Image Features (41K / 7G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1CmI6U8RsGuO9Rk8x61g8cAvghpa3ifxq/view?usp=sharing) 
- COCO 2015 Testing Image Features (81K / 13G) &ensp;   [Google Drive](https://drive.google.com/file/d/1Ua5lL5PvuuKnqWlqURdMqSnrpv7sQh2v/view?usp=sharing) 



#### 10-100 Updown Boxes

For users can extract VC Features if they want, here we also provide the Updown feature **box coordinates**:

- COCO 2014 Train/Val Image Boxes &ensp;   [Google Drive](https://drive.google.com/file/d/1BqLISOwaSdXngiiG_SeCgIE1CCOV2zHT/view?usp=sharing)
- COCO 2014 Testing Image Features &ensp;  [Google Drive](https://drive.google.com/file/d/17e_aRXO-1rZWQZm8RiL52h07y2prS3EO/view?usp=sharing)
- COCO 2015 Testing Image Features &ensp;   [Google Drive](https://drive.google.com/file/d/1TARtbsdFpiIxLRh91vJb_vuOdUmNiBGU/view?usp=sharing)



### How to use after download

- Unzip the file with command:
```bash
tar -xzvf file_name
```

- The feature format (The shape of each numpy file is [n x 1024]):
```
coco_trainval/test_year
 |---image_id1.npy
 |---image_id2.npy
  ...
 |---image_idN.npy
```

- Concatenate on the previous feature in the downstream task training.



### Downstream Vision & Language Tasks
Please check [Downstream Tasks](downstream/README.md) for more details:


**Some tips for using in downstream tasks**

- We recommend users to **add the dimension** of the start multi-layers (embedding layer, fc and so on) in the downstream networks since the feature size add from 2048 to 3072 (for Up-Down Feature).
- The learning rate can be **slighted reduced**.
- We find the **self-attentive operation** on feature (e.g., the refining encoder in AoANet) may hurt the effectiveness of our VC Feature. Details can be kindly found at the bottom of Page 7 in our paper.
- The concatenation of Up-Down and VC Feature **maybe too simple** for some downstream tasks. It can be regarded as a baseline and I believe there would be more potential on VC Feature. 

</br>

## VC R-CNN Framework

### Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions.



### Perform Training on COCO Dataset

#### Prepare Training Data

1. First, you need to download the COCO dataset and annotations. We assume that you save them in `/path_to_COCO_dataset/`
2. Then you need modify the path in `vc_rcnn/config/paths_catalog.py`, containing the `DATA_DIR` and `DATASETS path`.

#### Training Parameters

- `default.py`: `OUTPUT_DIR` denotes the model output dir. `TENSORBOARD_EXPERIMENT` is the tensorboard loger output dir. Another parameter the user may need notice is the `SOLVER.IMS_PER_BATCH` which denotes the number of total images per batch.
- Config file (e.g., `e2e_mask_rcnn_R_101_FPN_1x.yaml`): The main parameters the user may pay attention to is the training schedule and learning rate, and the used dataset.
- Parameters about VC: They are in the end of `default.py` with annotations. Users can make changes according to their own situation.

#### Running

Most of the configuration files that we provide assume that we are running 2 images on each GPU with 8 GPUs. In order to be able to run it on fewer GPUs, there are a few possibilities: 

**1. Single GPU Training:** 
Modify the cfg parameters. Here is an example:

```bash
python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_101_FPN_1x.yaml" --skip-test SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
```

Ps: To running more images on one GPU, you can refer to the [maskrcnn-benchmark].

**2. Multi-GPU training:**
The maskrcnn-benchmark directly support the multi-gpu training with `torch.distributed.launch`. You can run the command like (you need change $NGPUS to the num of GPU you use):

```bash
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "path/to/config/file.yaml" --skip-test MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN images_per_gpu x 1000
```

**Notes**: 

- In our experiments, we adopted `e2e_mask_rcnn_R_101_FPN_1x.yaml` **without the Mask Branch** (set False) as our config file.
- When training VC, actually we need not test scripts, thus we set `--skip-test` to skip the test process after training. The test script is used to extract vc feature. Or if you design your own test, you can remove `--skip-test`.
- The `MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN` denotes that the proposals are selected for per the batch rather than per image in the default training. The value is calculated by **1000 x images-per-gpu**. Here we have 2 images per GPU, therefore we set the number as 1000 x 2 = 2000. If we have 8 images per GPU, the value should be set as 8000. See [#672@maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/issues/672) for more details.
- Please note that the learning rate & iteration change rule follows the [scheduling rules from Detectron](https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14-L30), which means the lr need to be set 2x if the number of GPUs become 2x. In our methods, the learning rate is set for 4 GPUs and each GPU has 2 images.
- In my practice, the learning rate can not be best customized since the VC training is not a supervised model and you cannot measure the goodness of the VC model from training procedure. We have provide a general suitable learning rate and you can make some slight modification. 
- You can turn on the **Tensorboard** logger by add `--use-tensorboard` into command (Need to install tensorflow and tensorboardx first).
- The confounder dictionary `dic_coco.npy` and the prior `stat_prob.npy` are in the [tools](tools).



### Evaluation (Feature Extraction)

**1. Using your own model**

Since the goal of our VC R-CNN is to train the visual commonsense representations by self-supervised learning, we have no metrics for evaluation and we treat it as the feature extraction process.

Specifically, you can just run the following command to achieve the features.

```bash
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "path/to/config/file.yaml" TEST.IMS_PER_BATCH images_per_gpu x $GPUS
```

Please note that before running, you need to set the suitable path for `BOUNDINGBOX_FILE` and `FEATURE_SAVE_PATH` in `default.py`. (Recall that just given image and bounding box coordinate, our VC R-CNN can extract the VC Feature)


**2. Using our pretrained VC model on COCO**

Here we also provide our pretrained VC [model](https://drive.google.com/drive/folders/1y44pwGVVzRTr11tDKGnNEabOrif01QX4?usp=sharing). You can put it into the model dictionary and set the `last_checkpoint` with the absolute path of `model_final.pth`. Then run the command:

```bash
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "path/to/config/file.yaml" TEST.IMS_PER_BATCH images_per_gpu x $GPUS
```




### Add your Customized Dataset

**1. Training on customized dataset**

For learning VC Feature on your own dataset, the crux is to make your own dataset **COCO-style** (can refer to the data format in detection task) and design the dataloader file, for example `coco.py` and `openimages.py`. Here we provide an example for reference.


```python
from vc_rcnn.structures.bounding_box import BoxList

class MyDataset(object):
    def __init__(self, ...):
        # load the paths and image annotation files you will need in __getitem__

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes as a list of list of boxes.
        boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        labels = torch.tensor([10, 20])

        # create a BoxList from the boxes. Please pay attention to the box FORM (XYXY or XYWH or another)
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
		# Here you can also add many other characters to the boxlist in addition to the labels, for example `image_id', `category_id' and so on.
        
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": img_height, "width": img_width}
```

Then, you need modify the following files:

- [`vc_rcnn/data/datasets/__init__.py`](maskrcnn_benchmark/data/datasets/__init__.py): add it to `__all__`
- [`vc_rcnn/config/paths_catalog.py`](maskrcnn_benchmark/config/paths_catalog.py): `DatasetCatalog.DATASETS` and corresponding `if` clause in `DatasetCatalog.get()`



**2. Extracting features of customized dataset**

Recall that with the trained VC R-CNN, we can directly extract VC Features given raw images and bounding box coordinates. Therefore, the method to design dataloader is similar to the above. The only difference is you may want to load box coordinates file for feature extraction and the labels, classes is unnecessary.

You can also refer to `openimages.py` and `vcr.py`. 



**3. Some Tips and Traps**

- As our experiment results shown in paper, training our VC R-CNN on a larger dataset cannot bring much gain to the downstream tasks on other datasets. The probable reason maybe the COCO is enough to learn the commonsense feature for its downstream tasks. Therefore we **suggest** users: if you want to perform downstream tasks on Dataset A, you can firstly train our VC on the Dataset A. 

- When you design the Dataloader file, the most important thing is to pay attention to the box format (`XYXY` or `XYWH`) and adopt the correct command to load them. I have made this mistakes at the beginning of my project.

</br>

## Our experience you may need

Here we provide our experience (mainly the failure 2333) in training our VC R-CNN and using VC Feature. And we hope it can provide some help or possible ideas to the users to further develop this field :)

**1. For Training VC R-CNN**

- After reading the paper [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) by Kaiming He, I have tried to construct a better dictionary learning scheme for our VC Feature self-supervised learning. In our current implementation, the dictionary keep constant during training and we want to borrow the idea from MoCo to **UPDATE** the confounder dictionary iteratively. Since the iterative step (How often is the dictionary updated) can be set arbitrarily, I have tried a few steps but the result is similar. We want to further explore this in our future work.

- We have tried to add an 'Observation Window' into data sampling, which means for each image we just sample a window contains, for example 10 objects randomly each time. We want the model can learn the latent spatial relationship at the same time, however, the results can be worse. 

**2. For the VC Feature**

- As we discussed in our paper, our VC Feature achieves a less significant gain on **VQA** task than that for image captioning. We thought the possible reason can be the limited ability of the current question understanding. We are also wondering if we can train the **Vision & Language Commonsense Representations** in the future.
- In all the downstream tasks, we **just concatenate** the VC Feature on the previous Up-Down feature. This operation maybe too simple and I believe it does **NOT** reach VC 's full potential. My own bandwidth is limited but I know if more researchers try to use it and design more suitable downstream models, maybe we can create **more better results** :)
- The **evaluation** for the feature (self-supervised learning) can be too **trivial** and **hard**. This is the problem for all the self-supervised learning problem. The model performance **cannot** be estimated in training procedure. And if the things you want to learn is the feature (just like us), you need to evaluate it on many downstream tasks. Therefore, how to find a more effective way to evaluate the learning process can be a good point for research.


</br>

### Acknowledgement
I really appreciate [**Kaihua Tang**](https://kaihuatang.github.io/), **[Yulei Niu](https://yuleiniu.github.io/)**, **[Xu Yang](https://scholar.google.com.sg/citations?user=SqdxMH0AAAAJ&hl=zh-CN)**, **Jiaxin Qi**, **Xinting Hu** and [**Dong Zhang**](https://zhangdong-njust.github.io/) for their greatly helpful advice and lending me GPUs! 


If you have any questions or concerns, please kindly email to [**Tan Wang**](https://wangt-cn.github.io/).
