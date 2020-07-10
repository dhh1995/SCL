#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : model.py
# Author : Honghua Dong, Tony Wu
# Email  : dhh19951@gmail.com, tonywu0206@gmail.com
# Date   : 11/21/2019
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.logging import get_logger

from jactorch.quickstart.models import MLPModel

from analogy.constant import ORIGIN_IMAGE_SIZE
from analogy.nn.modules import ConvNet, ResNetWrapper, SharedGroupMLP
# from analogy.nn.symbolic.utils import transform

logger = get_logger(__file__)

__all__ = ['Visual2Symbolic']


class Visual2Symbolic(nn.Module):
    def __init__(self,
            conv_hidden_dims,
            output_dim,
            shared_group_mlp=True,
            input_dim=1,
            use_resnet=False,
            conv_repeats=None,
            conv_kernels=3,
            conv_residual_link=True,
            nr_visual_experts=1,
            mlp_hidden_dims=[],
            groups=1,
            split_channel=False,
            transformed_spatial_dim=None,
            mlp_transform_hidden_dims=[],
            image_size=ORIGIN_IMAGE_SIZE,
            use_layer_norm=False):
        super().__init__()
        if use_resnet:
            self.cnn = ResNetWrapper(
                repeats=conv_repeats,
                inplanes=conv_hidden_dims[0],
                channels=conv_hidden_dims,
                image_size=image_size)
        else:
            self.cnn = ConvNet(
                input_dim=input_dim,
                hidden_dims=conv_hidden_dims,
                repeats=conv_repeats,
                kernels=conv_kernels,
                residual_link=conv_residual_link,
                image_size=image_size,
                use_layer_norm=use_layer_norm)
        # self.cnn_output_size = self.cnn.output_size
        self.cnn_output_dim = self.cnn.output_dim
        h, w = self.cnn.output_image_size
        current_dim = h * w

        self.spatial_dim = current_dim
        self.transformed_spatial_dim = transformed_spatial_dim

        self.mlp_transform = None
        if transformed_spatial_dim is not None and transformed_spatial_dim > 0:
            self.mlp_transform = MLPModel(
                current_dim, transformed_spatial_dim,
                hidden_dims=mlp_transform_hidden_dims)
            current_dim = transformed_spatial_dim

        total_dim = self.cnn_output_dim * current_dim
        self.split_channel = split_channel
        if split_channel:
            current_dim = self.cnn_output_dim
        assert current_dim % groups == 0, ('the spatial dim {} should be '
            'divided by the number of groups {}').format(current_dim, groups)
        assert output_dim % (groups * nr_visual_experts) == 0, (
            'the output dim {} should be divided by the prod of number of '
            'groups {} and the number of visual experts {}').format(
                output_dim, groups, nr_visual_experts)

        self.shared_group_mlp = SharedGroupMLP(
            groups=groups,
            group_input_dim=total_dim // groups,
            group_output_dim=output_dim // (groups * nr_visual_experts),
            hidden_dims=mlp_hidden_dims,
            nr_mlps=nr_visual_experts,
            shared=shared_group_mlp)
        self.output_dim = output_dim

    def forward(self, x):
        x = x.float()
        nr_images, h, w = x.size()[1:]
        # x.shape: (batch, nr_img, h, w)
        x = x.view(-1, 1, h, w)
        # x.shape: (batch * nr_img, 1, h, w)
        x = self.cnn(x)
        # x.shape: (batch * nr_img, cnn_output_dim, h', w')
        current_dim = self.spatial_dim
        x = x.view(-1, current_dim)
        # x.shape: (batch * nr_img * cnn_output_dim, current_dim)
        if self.mlp_transform:
            x = self.mlp_transform(x)
            current_dim = self.transformed_spatial_dim
            # x.shape: (batch * nr_img * cnn_output_dim, current_dim)
        x = x.view(-1, self.cnn_output_dim, current_dim)
        # x.shape: (batch * nr_img, cnn_output_dim, current_dim)
        if self.split_channel:
            x = x.permute(0, 2, 1).contiguous()
            # x.shape: (batch * nr_img, current_dim, cnn_output_dim)
            current_dim = self.cnn_output_dim
        x = self.shared_group_mlp(x)
        # x.shape: (batch * nr_img, output_dim)
        x = x.view(-1, nr_images, self.output_dim)
        # x.shape: (batch, nr_img, output_dim)
        return x
