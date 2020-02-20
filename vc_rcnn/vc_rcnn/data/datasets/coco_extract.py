import torch
import torchvision
import json
import h5py
from PIL import Image
import os
from vc_rcnn.structures.bounding_box import BoxList


class OpenDataset(torchvision.datasets.coco.CocoDetection):

    def __init__(self, datadir, ann_path, causal_path, transforms=None):

        self.img_root = datadir
        self.transforms = transforms
        #self.image_pth = '/data3/wangtan/openimage/target_dir/coco_debug/images/train2017'


        with open(ann_path) as f:
            im_data = json.load(f)
        self.causal_file = h5py.File(causal_path, 'r')


        # self.img_height = [image['height'] for image in im_data['images']]
        # self.img_weight = [image['weight'] for image in im_data['images']]
        self.img_info = [image for image in im_data['images']]
        self.filenames = [os.path.basename(image['file_name']) for image in im_data['images']]
        # self.ann_file = [ann for ann in im_data['annotations']]

        #self.ids = sorted(self.ids)

        # self.categories = {cat['id']: cat['name'] for cat in im_data['categories']}
        # self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}


        # self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms


    def __getitem__(self, idx):

        img_path = os.path.join(self.img_root, self.filenames[idx])
        img = Image.open(img_path).convert("RGB")
        try:
            assert img.size[1] == self.img_info[idx]['height'] and img.size[0] == self.img_info[idx]['width']
        except AssertionError:
            print(self.filenames[idx])
        #causal_prob = self.causal_file[self.filenames[idx]]['causal_prob'][:]
        target = self.observ_get_groundtruth(idx)
        # img_id = self.filenames[idx]
        # img_orignal_size = img.size

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx


    def __len__(self):
        return len(self.filenames)


    def get_img_info(self, index):

        return self.img_info[index]


    def get_img_ann(self, index):
    # find the annotations for a given image
        image_ann = []
        for ann in self.ann_file:
            if ann['image_id'] == index:
                image_ann.append(ann)
        return image_ann
