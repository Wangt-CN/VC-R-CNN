import torch
import torchvision
import json
import h5py
from PIL import Image
import os
from vc_rcnn.structures.bounding_box import BoxList



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


class OpenDataset(torchvision.datasets.coco.CocoDetection):

    def __init__(self, datadir, ann_path, causal_path, transforms=None):

        self.img_root = datadir
        self.transforms = transforms


        self.img_info = json.load(open('/gruntdata3/wangtan/openimage/annotations/instances_tran2017_image.json', 'r'))
        # self.anno_info = [anno for anno in im_data['annotations']]
        self.anno_box = json.load(open('/gruntdata3/wangtan/openimage/annotations/instances_train2017_box.json', 'r'))
        self.anno_cat = json.load(open('/gruntdata3/wangtan/openimage/annotations/instances_train2017_cat.json', 'r'))
        self.filenames = [os.path.basename(image['file_name']) for image in self.img_info]
        self.id2cat = json.load(open('/gruntdata3/wangtan/openimage/annotations/instances_train2017_catlist.json', 'r'))
        self.categories = {cat['id']: cat['name'] for cat in self.id2cat}
        # self.ann_file = [ann for ann in im_data['annotations']]

        self._transforms = transforms


    def __getitem__(self, idx):

        img_path = os.path.join(self.img_root, self.filenames[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = self.anno_box[str(idx)]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = self.anno_cat[str(idx)]
        # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes) - 1
        target.add_field("labels", classes)

        try:
            assert img.size[1] == self.img_info[idx]['height'] and img.size[0] == self.img_info[idx]['width']
        except AssertionError:
            print(self.filenames[idx])

        w, h = img.size[0], img.size[1]
        sizes = [[w, h] for i in range(boxes.size(0))]
        sizes = torch.tensor(sizes)
        target.add_field("orignal_size", sizes)

        idxx = [idx for i in range(boxes.size(0))]
        idxx = torch.tensor(idxx)
        target.add_field("idx", idxx)

        target = target.clip_to_image(remove_empty=True)

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






