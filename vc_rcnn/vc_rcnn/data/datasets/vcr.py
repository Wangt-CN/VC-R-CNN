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

        # classes = self.anno_cat[str(idx)]
        # # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        # classes = torch.tensor(classes) - 1
        # target.add_field("labels", classes)

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


    def calculate_nb(self, boxes, classes, k):
        '''
        given the bbox list of an image (n*4), return the nearest k bbox index square
        :param boxes: the bbox list
        :param classes: the bbox list classes
        :param k: k nearest bbox
        :return:
        '''
        x = 0.5*(boxes[:, 0] + boxes[:, 2]).unsqueeze(1)
        y = 0.5*(boxes[:, 1] + boxes[:, 3]).unsqueeze(1)
        cor = torch.cat((x,y), 1)
        dis_matrix = self.l2_distance(cor[:,None,:], cor[None,:,:], 2)
        top_idx = torch.topk(dis_matrix, k=k, dim=1, largest=False)[1]
        nb = list(map(lambda x: classes[x], top_idx))
        nb = torch.stack(nb, 0)
        return nb[:,1,:], nb[:,2,:], nb[:,3,:]


    def l2_distance(self, x1, x2, dim):
        '''
        :param x1: vector x1
        :param x2: vector x2
        :param dim: the dimension of l2_distance
        '''
        return torch.pow(torch.sum((x1 - x2).pow(2), dim), 0.5)


    def observ_get_groundtruth(self, index):

        w, h = self.img_info[index]['width'], self.img_info[index]['height']


        # For openimage 501 classes

        # image_anns = self.get_img_ann(index)
        # boxes = [obj["bbox"] for obj in image_anns]
        # boxes = torch.as_tensor(boxes).reshape(-1, 4)
        # target = BoxList(boxes, (w, h), mode="xyxy")
        #
        # classes = [obj["category_id"] for obj in image_anns]
        # classes = torch.tensor(classes)
        # target.add_field("labels", classes)


        # For pretrained model on COCO 81 classes
        observ_bbox = self.causal_file[self.filenames[index]]['bbox'][:]
        boxes = [observ_bbox[idx] for idx in range(observ_bbox.shape[0])]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (w, h), mode="xyxy")

        classes = self.causal_file[self.filenames[index]]['class_label'][:]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        classes_soft = self.causal_file[self.filenames[index]]['soft_label'][:]
        classes_soft = torch.tensor(classes_soft)
        target.add_field("labels_soft", classes_soft)

        label_nb1, label_nb2, label_nb3 = self.calculate_nb(boxes, classes_soft, 4)
        # label_nb = torch.tensor(label_nb)
        target.add_field("labels_nb1", label_nb1)
        target.add_field("labels_nb2", label_nb2)
        target.add_field("labels_nb3", label_nb3)

        sizes = [[w, h] for idx in range(classes_soft.size(0))]
        sizes = torch.tensor(sizes)
        target.add_field("orignal_size", sizes)
        # target.add_field("image_id", self.filenames[index])

        target = target.clip_to_image(remove_empty=True)

        return target




    def observ_get_groundtruth_coco(self, index):

        w, h = self.img_info[index]['width'], self.img_info[index]['height']

        index_coco = 'tensor(' + str(self.img_info[index]['id']) + ')'

        # For pretrained model on COCO 81 classes
        observ_bbox = self.causal_file[index_coco]['bbox'][:]
        boxes = [observ_bbox[idx] for idx in range(observ_bbox.shape[0])]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (w, h), mode="xyxy")

        classes = self.causal_file[index_coco]['class_label'][:]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        classes_soft = self.causal_file[index_coco]['soft_label'][:]
        classes_soft = torch.tensor(classes_soft)
        target.add_field("labels_soft", classes_soft)

        label_nb1, label_nb2, label_nb3 = self.calculate_nb(boxes, classes_soft, 4)
        # label_nb = torch.tensor(label_nb)
        target.add_field("labels_nb1", label_nb1)
        target.add_field("labels_nb2", label_nb2)
        target.add_field("labels_nb3", label_nb3)

        sizes = [[w, h] for idx in range(classes_soft.size(0))]
        sizes = torch.tensor(sizes)
        target.add_field("orignal_size", sizes)
        # target.add_field("image_id", self.filenames[index])

        target = target.clip_to_image(remove_empty=True)

        return target