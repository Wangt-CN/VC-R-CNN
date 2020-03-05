#!/bin/bash

# # this installs the right pip and dependencies for the fresh python
conda install ipython pip
conda install python=3.7
# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python h5py lmdb
# install pytorch
conda install -c pytorch pytorch-nightly=1.0.0 torchvision=0.2.2
pip install 'pillow<7.0.0'

# install pycocotools
cd ..
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd ..
cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ..

# install VC
cd VC-R-CNN
python setup.py build develop
