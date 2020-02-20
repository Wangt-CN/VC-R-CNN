# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import numpy as np
import h5py
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .roi_box_predictors import make_causal_predictor
import torch.nn.functional as F
import pdb
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.causal_predictor = make_causal_predictor(cfg, self.feature_extractor.out_channels)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # if self.training:
        #     # Faster R-CNN subsamples during training the proposals with a fixed
        #     # positive / negative ratio
        #     with torch.no_grad():
        #         proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits = self.predictor(x)
        class_logits_causal_list, attn_list = self.causal_predictor(x, proposals)
        # pdb.set_trace()


        if not self.training:
            # result = self.post_processor((class_logits, box_regression), proposals)
            # self.save_object_feature_coco(x, class_logits, targets)
            result = self.post_processor_gt(x, class_logits, proposals)
            # result = self.post_processor_gt_attn(x, class_logits, attn_list, proposals)
            self.save_object_feature_gt_bu(x, result, targets)
            # self.save_object_attn_gt(x, result, targets)

            return x, result, {}

        loss_classifier, loss_causal = self.loss_evaluator(
            [class_logits], class_logits_causal_list, proposals
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_causal=loss_causal),
        )

    def post_processor_gt(self, x, class_logits, boxes):
        class_prob = F.softmax(class_logits, -1)
        bbx_idx = torch.arange(0, class_logits.size(0))
        # image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        class_prob = class_prob.split(boxes_per_image, dim=0)
        bbx_idx = bbx_idx.split(boxes_per_image, dim=0)

        for i, (class_prob_image, bbx_idx_image) in enumerate(zip(class_prob, bbx_idx)):
            boxes[i].add_field("labels_classify", torch.max(class_prob_image, 1)[0])
            boxes[i].add_field("features", x[bbx_idx_image])

        return boxes

    def post_processor_gt_attn(self, x, class_logits, attn_list, boxes):
        # class_prob = F.softmax(class_logits, -1)
        bbx_idx = torch.arange(0, class_logits.size(0))
        # image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        # class_prob = class_prob.split(boxes_per_image, dim=0)
        bbx_idx = bbx_idx.split(boxes_per_image, dim=0)

        for i, (attn_image, bbx_idx_image) in enumerate(zip(attn_list, bbx_idx)):
            boxes[i].add_field("attn", attn_image)
            boxes[i].add_field("features", x[bbx_idx_image])

        return boxes

    def save_object_feature_gt_bu(self, x, result, targets):

        for i, image in enumerate(result):
            feature_pre_image = image.get_field("features").cpu().numpy()
            try:
                assert image.get_field("num_box")[0] == feature_pre_image.shape[0]
                image_id = str(image.get_field("image_id")[0].cpu().numpy())
                path = '/data4/vc/vc-rcnn-onlyy/vc-rcnn-onlyy/' + image_id +'.npy'
                np.save(path, feature_pre_image)
            except:
                print(image)

    def save_object_attn_gt(self, x, result, targets):
        gpu_id = x.get_device()

        with h5py.File('/data2/tingjia/wt/openimage/target_dir/coco/vc-feature/attn' + str(gpu_id) + '.hdf5', 'a') as f:
            for i, image in enumerate(result):
                # idx_pre_image = image.get_field("idx")
                class_gtlabel_per_image = image.get_field("labels")
                attn_pre_image = image.get_field("attn")
                # class_label_per_image = image.get_field("labels_classify")

                # feature_pre_image = image.get_field("features")
                # if class_gtlabel_per_image.size(0) >= 1 and targets[i].get_field("image_id").size(0) > 0:
                    # image_id_h5py = f.create_group(img_pths[i])

                image_id_h5py = f.create_group(str(targets[i].get_field("image_id")[0].cpu().numpy()))
                image_id_h5py.create_dataset("attn", data=attn_pre_image.cpu())
                image_id_h5py.create_dataset("class_gtlabel", data=class_gtlabel_per_image.cpu())
                # image_id_h5py.create_dataset("class_label", data=class_label_per_image.cpu())

                original_size = targets[i].get_field("orignal_size")[0]
                image = image.resize((original_size[0], original_size[1]))

                image_id_h5py.create_dataset("bbox", data=image.bbox.cpu())


    def save_object_feature_gt(self, x, result, targets):
        gpu_id = x.get_device()

        with h5py.File('/data2/tingjia/wt/openimage/target_dir/coco/vc-feature/coco_test_all_vc_xy_10_100_' + str(gpu_id) + '.hdf5', 'a') as f:
            for i, image in enumerate(result):
                # idx_pre_image = image.get_field("idx")
                class_gtlabel_per_image = image.get_field("labels")
                # class_label_per_image = image.get_field("labels_classify")

                feature_pre_image = image.get_field("features")
                if class_gtlabel_per_image.size(0) >= 1 and targets[i].get_field("image_id").size(0) > 0:
                    # image_id_h5py = f.create_group(img_pths[i])
                    if str(targets[i].get_field("image_id")[0].cpu().numpy()) in f:
                        del f[str(targets[i].get_field("image_id")[0].cpu().numpy())]
                    image_id_h5py = f.create_group(str(targets[i].get_field("image_id")[0].cpu().numpy()))
                    image_id_h5py.create_dataset("feature", data=feature_pre_image.cpu())
                    image_id_h5py.create_dataset("class_gtlabel", data=class_gtlabel_per_image.cpu())
                    # image_id_h5py.create_dataset("class_label", data=class_label_per_image.cpu())

                    original_size = targets[i].get_field("orignal_size")[0]
                    image = image.resize((original_size[0], original_size[1]))

                    image_id_h5py.create_dataset("bbox", data=image.bbox.cpu())


    def save_object_feature_coco(self, x, result, targets):
        gpu_id = x.get_device()
        if gpu_id == 0:
            with h5py.File('/data2/tingjia/wt/openimage/target_dir/Openimages/coco_vctrain/coco_train_all_vc2.hdf5', 'a') as f:
                for i, image in enumerate(result):
                    idx_pre_image = image.get_field("idx")
                    softscore_pre_image = image.get_field("soft_scores")
                    class_label_per_image = image.get_field("labels")

                    feature_pre_image = x[idx_pre_image]
                    if class_label_per_image.size(0) >= 1 and targets[i].get_field("image_id").size(0) > 0:

                        # image_id_h5py = f.create_group(img_pths[i])
                        # exist_id = str(targets[i].get_field("image_id")[0].cpu().numpy())
                        # if exist_id not in f.keys():
                        image_id_h5py = f.create_group(str(targets[i].get_field("image_id")[0].cpu().numpy()))
                        image_id_h5py.create_dataset("feature", data=feature_pre_image.cpu())
                        image_id_h5py.create_dataset("soft_label", data=softscore_pre_image.cpu())
                        image_id_h5py.create_dataset("class_label", data=class_label_per_image.cpu())

                        original_size = targets[i].get_field("orignal_size")[0]
                        image = image.resize((original_size[0], original_size[1]))
                        image_id_h5py.create_dataset("bbox", data=image.bbox.cpu())



def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
