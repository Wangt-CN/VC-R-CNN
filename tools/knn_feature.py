import h5py
import numpy as np
import torch


coco_rcnn = h5py.File('/data3/wangtan/openimage/target_dir/coco/coco_train_all_gt.hdf5','r')
coco_vc = h5py.File('/data3/wangtan/openimage/target_dir/coco/coco_train_all_vc_0.hdf5', 'r')


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def l2_sim(x1, x2, dim=1):
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)

    return torch.pow(torch.sum((x1 - x2).pow(2), dim), 0.5)


for image_id in coco_vc.keys():
    feature_rcnn = coco_rcnn[image_id]['feature'][:]
    feature_vc = coco_vc[image_id]['feature'][:]
    sim_rcnn = cosine_sim(feature_rcnn[:, None, :], feature_rcnn[None, :, :])
    sim_vc = cosine_sim(feature_vc[:, None, :], feature_vc[None, : , :])