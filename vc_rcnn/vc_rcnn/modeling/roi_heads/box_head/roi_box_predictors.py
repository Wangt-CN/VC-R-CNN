# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from vc_rcnn.modeling import registry
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)

        nn.init.normal_(self.cls_score.weight, std=0.01)

        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)

        return scores


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)



@registry.ROI_BOX_PREDICTOR.register("CausalPredictor")
class CausalPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CausalPredictor, self).__init__()

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.embedding_size = cfg.MODEL.ROI_BOX_HEAD.EMBEDDING
        representation_size = in_channels

        self.causal_score = nn.Linear(2*representation_size, num_classes)
        self.Wy = nn.Linear(representation_size, self.embedding_size)
        self.Wz = nn.Linear(representation_size, self.embedding_size)

        nn.init.normal_(self.causal_score.weight, std=0.01)
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)
        nn.init.constant_(self.causal_score.bias, 0)

        self.feature_size = representation_size
        self.dic = torch.tensor(np.load(cfg.DIC_FILE)[1:], dtype=torch.float)
        self.prior = torch.tensor(np.load(cfg.PRIOR_PROB), dtype=torch.float)

    def forward(self, x, proposals):
        device = x.get_device()
        dic_z = self.dic.to(device)
        prior = self.prior.to(device)

        box_size_list = [proposal.bbox.size(0) for proposal in proposals]
        feature_split = x.split(box_size_list)
        xzs = [self.z_dic(feature_pre_obj, dic_z, prior) for feature_pre_obj in feature_split]

        causal_logits_list = [self.causal_score(xz) for xz in xzs]


        return causal_logits_list


    def z_dic(self, y, dic_z, prior):
        """
        Please note that we computer the intervention in the whole batch rather than for one object in the main paper.
        """
        length = y.size(0)

        attention = torch.mm(self.Wy(y), self.Wz(dic_z).t()) / (self.embedding_size ** 0.5)
        attention = F.softmax(attention, 1)
        z_hat = attention.unsqueeze(2) * dic_z.unsqueeze(0)
        z = torch.matmul(prior.unsqueeze(0), z_hat).squeeze(1)
        xz = torch.cat((y.unsqueeze(1).repeat(1, length, 1), z.unsqueeze(0).repeat(length, 1, 1)), 2).view(-1, 2*y.size(1))

        # detect if encounter nan
        if torch.isnan(xz).sum():
            print(xz)
        return xz

def make_causal_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR["CausalPredictor"]
    return func(cfg, in_channels)
