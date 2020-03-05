import torch
import torchvision
import json
import h5py
from PIL import Image
import os
from vc_rcnn.structures.bounding_box import BoxList
import lmdb
import numpy as np
import base64
import pickle

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class vcrDataset(torchvision.datasets.coco.CocoDetection):

    def __init__(self, datadir, ann_file, transforms=None):

        self.img_root = datadir
        self.transforms = transforms
        #self.image_pth = '/data3/wangtan/openimage/target_dir/coco_debug/images/train2017'



        # self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        features_path = '/data3/wangtan/vc/vilbert_beta/data/VCR/VCR_gt_resnet101_faster_rcnn_genome.lmdb'
        self.env = lmdb.open(features_path, max_readers=1, readonly=True,
                            lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get('keys'.encode()))
            self.img_info = []
            for i in self._image_ids:
                # a = pickle.loads(txn.get(i))
                h = pickle.loads(txn.get(i))['image_h']
                w = pickle.loads(txn.get(i))['image_w']
                path = pickle.loads(txn.get(i))['image_id']
                self.img_info.append({"width":w, "height":h, "path":path})



    def __getitem__(self, idx):

        image_id = self._image_ids[idx]
        img_name = self.img_info[idx]['path']
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert("RGB")

        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(image_id))
            image_id_ = item['image_id']
            image_h = int(item['image_h'])
            image_w = int(item['image_w'])
            num_boxes = int(item['num_boxes'])
            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)


        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        try:
            assert img.size[1] == image_h and img.size[0] == image_w
        except AssertionError:
            print(image_id)

        w, h = img.size[0], img.size[1]
        sizes = [[w, h] for i in range(boxes.size(0))]
        sizes = torch.tensor(sizes)
        target.add_field("orignal_size", sizes)

        image_id_all = [int(image_id) for i in range(boxes.size(0))]
        image_id_all = torch.tensor(image_id_all)
        target.add_field("image_id", image_id_all)

        numm = [num_boxes for i in range(boxes.size(0))]
        numm = torch.tensor(numm)
        target.add_field("num_box", numm)

        target = target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx


    def __len__(self):
        return len(self._image_ids)


    def get_img_info(self, index):

        return self.img_info[index]


    def get_img_ann(self, index):
    # find the annotations for a given image
        image_ann = []
        for ann in self.ann_file:
            if ann['image_id'] == index:
                image_ann.append(ann)
        return image_ann


