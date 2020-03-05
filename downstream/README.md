# Downstream Tasks
With our proposed VC Feature, we have achieved many new SOTA results in Vision & Language downstream tasks. Since the usage of our feature is very easy, here we just provide the code for reference. The first choice is still visit the original repo of models :)

Please **NOTE** that what we do is just to **concatenate** our VC Feature on the previous feature and make some slight adjustments on parameters, which means our VC Feature can be quite **general** to use in many other tasks and models. We are very **welcome** and **appreciate** users to try our VC Feature on **other tasks** (current SOTA models) and **share results/experience** or directly **pull request**, especially on other Vision & Language tasks (such as Image-text matching, Scene-graph and so on). 

If any questions or problems, also welcome discussion. Let us pursing better results together. Thanks to all the opensource code!




## Image Captioning



### Up-Down Model
**Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering** 
[[**Paper**]](https://arxiv.org/abs/1707.07998) [[**github**]](https://github.com/peteanderson80/Up-Down-Captioner)


For the classical Up-Down model, we use the well-known [codebase](https://github.com/ruotianluo/self-critical.pytorch) by [Ruotian Luo](https://github.com/ruotianluo).

Compared to the orignal repo, we just modify the [`dataloader.py`]() and the parameter setting to the [`opt.py`]() to support our VC Feature. Therefore for the ''old driver''(hhh) who also uses Ruotian Luo's code, you can continue following his code and just replace the file `dataloader.py` or modified the code by yourself.
For those who is a newbie, you can just git download this repo and follow the command below to start training:

#### Requirements

- Python 2.7
- Java 1.8.0
- PyTorch 0.4.1

#### Prepare Data
Please refer to [here](https://github.com/ruotianluo/self-critical.pytorch#prepare-data)

#### Start training
```bash
$ python train.py --id topdown --caption_model topdown --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_att_dir_vc [the/path/to/VC_Feature/trainval] --input_att_dir [the/path/to/Updown_Feature] --batch_size 50 --learning_rate 3e-4 --checkpoint_path log_topdown_lr_3 --save_checkpoint_every 2200 --val_images_use 5000 --max_epochs 80 --rnn_size 2048 --input_encoding_size 1024 --self_critical_after 30 --language_eval 1 --learning_rate_decay_start 0 --scheduled_sampling_start 0
```

NOTE: This command mix the cross-entropy and self-critical training. If you want to training them separately, you may need:


#### Cross Entropy Training
```bash
$ python train.py --id topdown --caption_model topdown --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_att_dir_vc [the/path/to/VC_Feature/trainval] --input_att_dir [the/path/to/Updown_Feature] --batch_size 50 --learning_rate 3e-4 --checkpoint_path log_topdown --save_checkpoint_every 2200 --val_images_use 5000 --rnn_size 2048 --input_encoding_size 1024 --max_epochs 30 --language_eval 1
```
#### Self-critical Training
```bash
$ python train.py --id topdown --caption_model topdown --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_att_dir_vc [the/path/to/VC_Feature/trainval] --input_att_dir [the/path/to/Updown_Feature]  --batch_size 50 --learning_rate 3e-5 --start_from log_topdown --checkpoint_path log_topdown --save_checkpoint_every 2200 --language_eval 1 --val_images_use 5000 --self_critical_after 30 --rnn_size 2048 --input_encoding_size 1024 --cached_tokens coco-train-idxs --max_epoch 80
```

#### Evaluation
```bash
python eval.py --model log_topdown/model-best.pth --infos_path log_topdown/infos_topdown-best.pkl  --dump_images 0 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 50 --split test
```

Ps: the repo of Ruotian Luo also contains some other Image captioning methods, which can be convenient for users to directly try our feature on them.

</br>

### AoANet

**Attention on Attention for Image Captioning** 
[[**Paper**]](https://arxiv.org/abs/1908.06954) [[**github**]](https://github.com/husthuaan/AoANet)

Compared to the original AoANet [codebase](https://github.com/husthuaan/AoANet) by [Lun Huang](https://husthuaan.github.io/) , we make the following change:

- Concentrate on our VC Feature (`dataloader.py`, `train.sh`)
- Discard the AoANet encoder refining module (`train.sh`)
- Change some parameters in `train.sh`

And that's all! We have got **[Cider 128.1](https://competitions.codalab.org/my/competition/submission/514626/stdout.txt)** which is the SOTA single captioning model by 11/16/2019. Here we also upload our used code for reference and you can also compare them with the original code. 

#### Requirements
- Python 3.6
- Java 1.8.0
- PyTorch 1.0

#### Prepare Data
Please refer to [here](https://github.com/husthuaan/AoANet#prepare-data)

#### Start training
```bash
CUDA_VISIBLE_DEVICES=0 sh train.sh
```
NOTE: we modify parameters in `train.sh`

#### Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --model log/log_aoanet_rl/model-best.pth --infos_path log/log_aoanet_rl/infos_aoanet-best.pkl  --dump_images 0 --dump_json 1 --num_images -1 --language_eval 1 --beam_size 2 --batch_size 50 --split test
```


</br>


## VQA

As we wrote in our paper, we found that the performance gain of our VC Feature on VQA can be **slightly lower** than that in image captioning. We thought the probable reason maybe the lack of the understanding ability of our VC on the textual sentences. And we thought that maybe some **customized** architecture for the Up-Down+VC feature can be more effective. Welcome to discuss together.



### Up-Down model
**Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering** 
[[**Paper**]](https://arxiv.org/abs/1707.07998) [[**github**]](https://github.com/peteanderson80/Up-Down-Captioner)

For the Up-Down model in Visual Question Answering, we adopted the [codebase](https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch) by [Kaihua Tang](https://kaihuatang.github.io/). Similarly, this codebase contains some other methods in VQA, which can also be used with our VC Feature.

The original repo is very detailed and we just introduce the changes when we use:

- We concatenate our VC Feature on the previous Up-Down features, change the size of 2048 to 3072. (`config.py`)

- We change the initial learning rate to 2e-3 (`config.py`)

- The size of the attention and classifier has been changed to 2048

- Please note that the data this codebase use is the `hdf5` format, which means you may need to change the numpy to hdf5. We have provided the  convert code in `tools` document.



### MCAN

**Deep Modular Co-Attention Networks**
[[**Paper**]](http://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.html) [[**github**]](https://github.com/MILVLG/mcan-vqa)

 With the MCAN model, we have got the SOTA results ([Overall **71.21** on test-dev, **71.49** on test-std](https://evalai.s3.amazonaws.com/media/submission_files/submission_46389/6a17824b-cf60-4a66-b677-5bb10a138f5a.json)) with single model by 11/16/2019.

Here we directly provide the code and parameters training on the train+val set or you can also refer to the original MCAN [repo](https://github.com/MILVLG/mcan-vqa). The modification we made:

- We concatenate our VC Feature on the previous Up-Down features, change the size of 2048 to 3072. (`load_data.py`)
- The size of the `FLAT_MLP_SIZE` has been changed to 1024

#### Prepare Data
Please refer to [here](https://github.com/MILVLG/mcan-vqa#prerequisites). Moreover, since there are a little difference of few Up-Down feature samples in the original repos, I directly replace them with the Up-Down feature with numpy format. (I cannot make sure the latent reason).

#### Start Training
```bash
python3 run.py --RUN='train' --MODEL='large'
```

#### Online Evaluation
```bash
python3 run.py --RUN='test' --CKPT_V=str --CKPT_E=int
```

</br>

## Visual Commonsense Reasoning



### R2C

**From Recognition to Cognition: Visual Commonsense Reasoning**
[[**Paper**]](https://visualcommonsense.com/) [[**github**]](https://github.com/rowanz/r2c)

The original R2C model integrate the ResNet network into the model for feature extraction. Therefore, to make the Up-Down feature available, we discarded the ResNet and utilized the Up-Down feature extracted from [ViLBERT](https://github.com/jiasenlu/vilbert_beta) (The lmdb feature file can be downloaded from that repo).
Here we provide the detailed modification for reference and the code for training R2C with our VC Feature.

- We added a new file `_image_features_reader.py` containing reading Up-Down and VC features.

- Add the code about the Up-Down feature loader in `vcr.py`, `model.py`.

- Increase the layer size and modify the learning rate in `default.json`

- Note that you need to modify the data path in `vcr.py` and `_image_features_reader.py`

#### Start Training 
```bash
python train.py -params models/multiatt/default.json -folder /the/path/you/want/to/save
```
The detailed environment setting and data prepare, please refer to the original [repo](https://github.com/rowanz/r2c). If many users require the VC Feature on the VCR Dataset, we would release that feature. Or users can also train the VC Feature by theirselves.



### ViLBERT (To be updated)
**Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks**
[[**Paper**]](https://arxiv.org/abs/1908.02265) [[**github**]](https://github.com/facebookresearch/vilbert-multi-task)

We have noticed that ViLBERT has update their code, therefore we would re-run the code and refresh our results.



