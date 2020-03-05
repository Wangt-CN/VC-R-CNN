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
import os
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
        self.feature_save_path = cfg.FEATURE_SAVE_PATH

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


        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)

        # self predictor
        class_logits = self.predictor(x)

        # context predictor
        class_logits_causal_list = self.causal_predictor(x, proposals)
        # pdb.set_trace()


        if not self.training:

            result = self.post_processor_gt(x, class_logits, proposals)

            # save object feature
            self.save_object_feature_gt_bu(x, result, targets)

            return x, result, {}

        loss_classifier, loss_causal = self.loss_evaluator(
            [class_logits], class_logits_causal_list, proposals
        )
        return (
            x,
            proposals,
            dict(loss_self=loss_classifier, loss_causal=loss_causal),
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


    def save_object_feature_gt_bu(self, x, result, targets):

        for i, image in enumerate(result):
            feature_pre_image = image.get_field("features").cpu().numpy()
            try:
                assert image.get_field("num_box")[0] == feature_pre_image.shape[0]
                image_id = str(image.get_field("image_id")[0].cpu().numpy())
                path = os.path.join(self.feature_save_path, image_id) +'.npy'
                np.save(path, feature_pre_image)
            except:
                print(image)




def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
