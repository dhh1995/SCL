#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : observer.py
# Author : Honghua Dong, Tony Wu
# Email  : dhh19951@gmail.com, tonywu0206@gmail.com
# Date   : 11/27/2019
#
# Distributed under terms of the MIT license.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.logging import get_logger

from jactorch.quickstart.models import MLPModel

from ..constant import ORIGIN_IMAGE_SIZE, MAX_VALUE

logger = get_logger(__file__)

__all__ = ['Observer']

"""
    Align the learned representation with ground truth.

    For each independent features, learn a linear transformation for each ground
    truth features, minimize the L2 distance at all data-points. For each ground
    truth features, pick the one minimize the distance.
"""
class Observer(nn.Module):
    def __init__(self, input_dim, output_dim, feature_embedding_dim=1,
            exclude_angle_attr=False, key_attr_only=False, thresh=0.05,
            exclude_outside_color=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_embedding_dim = feature_embedding_dim
        self.exclude_angle_attr = exclude_angle_attr
        self.key_attr_only = key_attr_only # only count loss on color/size/type
        self.exclude_outside_color = exclude_outside_color
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, input_dim, output_dim, feature_embedding_dim))
        self.bias = nn.Parameter(torch.Tensor(1, 1, input_dim, output_dim))
        self.thresh = thresh
        self.restart()

    def restart(self):
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        monitors, output_dict = {}, {}
        pred = x['pred']
        # pred.shape: [batch, nr_image, input_dim * fed]
        label = x['label']
        label = label.float()
        # label.shape: [batch, nr_image, nr_parts, nr_attrs]
        fed = self.feature_embedding_dim
        batch, nr_images, nr_parts, nr_attrs = label.size()

        pred = pred.view(-1, nr_images, self.input_dim, 1, fed)
        # pred.shape: [batch, nr_image, input_dim, 1, fed]
        pred = pred.repeat(1, 1, 1, self.output_dim, 1)
        # pred.shape: [batch, nr_image, input_dim, output_dim, fed]
        label = label.view(-1, nr_images, 1, self.output_dim)
        # label.shape: [batch, nr_image, 1, output_dim]
        label = label.repeat(1, 1, self.input_dim, 1)
        # label.shape: [batch, nr_image, input_dim, output_dim]
        regression = (self.weight * pred).sum(dim=-1) + self.bias
        # regression.shape: [batch, nr_image, input_dim, output_dim]
        origin_label = label
        weight = self.weight[0, 0, :, :, 0]
        if self.exclude_angle_attr or self.key_attr_only:
            flag = np.zeros([nr_parts, nr_attrs])
            flag[:, 0] = 1 # the angle attr is index 0 in nr_attrs dim
            if self.key_attr_only:
                for i in range(4, nr_attrs):
                    flag[:, i] = 1
            flag = flag.reshape(-1)
            mask = np.where(flag == 0)[0]
            regression = regression[:, :, :, mask]
            label = label[:, :, :, mask]
            weight = weight[:, mask]

            if self.key_attr_only and nr_attrs == 7:
                exists = origin_label[:, :, :, 6::7]
                mask = exists.unsqueeze(-1)
                mask = mask.repeat(1, 1, 1, 1, 3).flatten(3, -1)
                label *= mask
                regression *= mask

        loss = (regression - label) ** 2

        avg_loss = loss.mean(dim=0).mean(dim=0)
        inds_add = 0
        if self.exclude_outside_color:
            # For in-out / in-distri, exclude the outside.color
            # which is always constant.
            inds_add = 1
            avg_loss = avg_loss[:, 1:]
        errs, inds = avg_loss.min(dim=1)
        output_dict['errs'] = errs
        output_dict['inds'] = inds + inds_add
        output_dict['weight'] = weight

        stat = avg_loss.min(dim=0)[0].mean()
        equal = label.eq(regression.round()).float()
        diff = (regression.round() - regression).abs()
        # [NOTE]: The acc will be higher when key_attr_only,
        # the masked out ones are counted into acc.
        acc = equal * (diff < self.thresh).float()
        acc = acc.mean(dim=0).mean(dim=0).max(dim=0)[0].mean()
        monitors['stat'] = stat
        monitors['acc'] = acc
        loss = loss.mean()
        return loss, monitors, output_dict
