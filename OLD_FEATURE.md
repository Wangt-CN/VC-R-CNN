## Old Feature

This is the old version of our VC Feature. The performance of this feature is slightly lower in Captioning and VQA Updown model. The gap is small and you can also have a try to utilize it.



#### 10 - 100 features per image (adaptive):

- COCO 2014 Train/Val Image Features (123K / 6G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1z36lR-CwLjwsJPPxE-phqZTMQCTb5KLV/view?usp=sharing)  &ensp;  [Baidu Drive (key:ec8x)](https://pan.baidu.com/s/1alOZkyGCJSso_znc2i2REA)
- COCO 2014 Testing Image Features (41K / 2G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1PQANKKRdD6j980SjokNTCXV5aQNGW4zS/view?usp=sharing)  &ensp;  [Baidu Drive (key:ec8x)](https://pan.baidu.com/s/1alOZkyGCJSso_znc2i2REA)
- COCO 2015 Testing Image Features (81K / 4G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1U9-EbQI8ZCFe7MvmJXI1xCDTbW2f-E98/view?usp=sharing)  &ensp;  [Baidu Drive (key:ec8x)](https://pan.baidu.com/s/1alOZkyGCJSso_znc2i2REA)

**Ps**: For those who may have no access to the Up-Down feature, here we also provide the feature **after concatenation** and you can directly use without `numpy.concatenate` (The feature dimension is 3072 : 2048+1024):

- [concat] COCO 2014 Train/Val Image Features (123K / 27G) &ensp;   [Google Drive](https://drive.google.com/file/d/1kBnVvph5ISWWljOPeWCdFHWg7lkOp6QX/view?usp=sharing)  
- [concat] COCO 2014 Testing Image Features (41K / 9G)  &ensp;  [Google Drive](https://drive.google.com/file/d/1dSx4BeUJT1NOW6Fdlnmo_3HEbv7zrp1B/view?usp=sharing) 
- [concat] COCO 2015 Testing Image Features (81K / 17G) &ensp;   [Google Drive](https://drive.google.com/file/d/1Sp8w8BTyiVMJjlSUJWvgFFTaH2AAayJQ/view?usp=sharing) 