# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py#L333
(Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


@weighted_loss
def bub_loss(pred,
            target,
            valid_mask,
            gamma=4/3,
            smooth=1e-10,
            class_weight=None,
            ignore_index=255):

    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            bub_loss = binary_bub_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                gamma=gamma,
                smooth=smooth)
            if class_weight is not None:
                bub_loss *= class_weight[i]
            total_loss += bub_loss
    return total_loss / num_classes


@weighted_loss
def binary_bub_loss(pred,
                    target,
                    valid_mask,
                    gamma=4/3,
                    smooth=1e-10):

    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) + smooth
    TN = torch.sum(torch.mul(1 - pred, 1 - target) * valid_mask, dim=1) + smooth
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)

    _TN = torch.sqrt(TP*TN)

    # l1 = (FP + FN)/(TP + FP + FN + _TN)
    bub = (TP +_TN) / (TP + FP + FN + _TN)
    bub_gamma = torch.pow(bub,gamma)
    _bub_gamma = torch.pow(1-bub,gamma)
    # l2 = (FP + FN) / (TN + FP + FN + _TN)
    # l2_gamma = torch.pow(l2, gamma)
    # _l2_gamma = torch.pow(1-l2,gamma)

    bub_v2 = bub_gamma / torch.pow(bub_gamma + _bub_gamma, 1/gamma)
    # tk_bub_v2 = l2_gamma / torch.pow((l2_gamma + _l2_gamma), 1/gamma)

    return bub_v2

@LOSSES.register_module()
class BUBLoss(nn.Module):
    def __init__(self,
                 smooth=1e-10,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 gamma = 4/3,
                 loss_name='loss_bub'):
        super(BUBLoss, self).__init__()
        self.smooth = smooth
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.gamma = gamma
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * bub_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            gamma=self.gamma,
            smooth=self.smooth,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
