#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : modules.py
# Author : Honghua Dong, Tony Wu
# Email  : dhh19951@gmail.com, tonywu0206@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

from jacinle.logging import get_logger

from jactorch.nn import Conv2d
from jactorch.quickstart.models import MLPModel

from analogy.constant import ORIGIN_IMAGE_SIZE

logger = get_logger(__file__)

__all__ = ['FCResBlock', 'Expert', 'Scorer', 'ConvBlock', 'ConvNet',
    'ResNet', 'ResNetWrapper']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FCResBlock(nn.Module):
    def __init__(self, nn_dim, use_layer_norm=True,):
        self.use_layer_norm = use_layer_norm
        super(FCResBlock, self).__init__()
        self.norm_in = nn.LayerNorm(nn_dim)
        self.norm_out = nn.LayerNorm(nn_dim)
        self.transform1 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform1.weight, std=0.005)
        self.transform2 = torch.nn.Linear(nn_dim, nn_dim)
        torch.nn.init.normal_(self.transform2.weight, std=0.005)

    def forward(self, x):
        if self.use_layer_norm:
            x_branch = self.norm_in(x)
        else:
            x_branch = x
        x_branch = self.transform1(F.relu(x_branch))
        if self.use_layer_norm:
            x_branch = self.norm_out(x_branch)
        x_out = x + self.transform2(F.relu(x_branch))
        #x_out = self.transform2(F.relu(x_branch))
        #x_out = F.relu(self.transform2(x_branch))
        #return F.relu(x_out)
        return x_out


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.mlp = MLPModel(input_dim, 1, hidden_dims=hidden_dims)

    def forward(self, x):
        return self.mlp(x)


"""
  The reduction group meaning the axis to be reduced together, in the order of
  (nr_fetures, nr_experts), from end to the front.

  groups = [2] means reduce all two once.
  groups = [1, 1] means first reduce nr_experts, then reduce nr_features
"""
class Scorer(nn.Module):
    def __init__(self, nr_features, nr_experts,
            hidden_dims=[], reduction_groups=[2], sum_as_reduction=0):
        super().__init__()
        assert sum(reduction_groups) == 2
        if len(reduction_groups) == 1 and sum_as_reduction > 0:
            logger.warning('The scorer is just a summation.')
        dims = [nr_features, nr_experts]
        self.nr_reductions = len(reduction_groups)
        self.sum_as_reduction = sum_as_reduction
        self.input_dims = []
        mlps = []
        for g in reduction_groups:
            input_dim = np.prod(dims[-g:])
            output_dim = 1
            dims = dims[:-g]
            mlp = MLPModel(input_dim, output_dim, hidden_dims=hidden_dims)
            self.input_dims.append(input_dim)
            mlps.append(mlp)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, x):
        # x.shape: (batch * nr_candidates, nr_features, nr_experts, num)
        for i in range(self.nr_reductions):
            x = x.view(-1, self.input_dims[i])
            s = self.sum_as_reduction
            if s is not None and i + s >= self.nr_reductions:
                x = x.sum(dim=-1)
            else:
                x = self.mlps[i](x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, h, w, repeats=1,
            kernel_size=3, padding=1, residual_link=True, use_layer_norm=False):
        super().__init__()
        convs = []
        norms = []
        if type(kernel_size) is int:
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size

        current_dim = input_dim
        for i in range(repeats):
            stride = 1
            if i == 0:
                # The reduction conv
                stride = 2
                h = (h + 2 * padding - kh + stride) // stride
                w = (w + 2 * padding - kw + stride) // stride
            convs.append(Conv2d(current_dim, output_dim,
                kernel_size=kernel_size, stride=stride, padding=padding))
            current_dim = output_dim
            if use_layer_norm:
                norms.append(nn.LayerNorm([current_dim, h, w]))
            else:
                norms.append(nn.BatchNorm2d(current_dim))

        self.residual_link = residual_link
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.output_size = (h, w)

    def forward(self, x):
        is_reduction = True
        for conv, norm in zip(self.convs, self.norms):
            # ConvNormReLU
            _ = x
            _ = conv(_)
            _ = norm(_)
            _ = F.relu(_)
            if is_reduction or not self.residual_link:
                x = _
            else:
                x = x + _
            is_reduction = False

        return x


class ConvNet(nn.Module):
    def __init__(self,
            input_dim,
            hidden_dims,
            repeats=None,
            kernels=3,
            residual_link=True,
            image_size=ORIGIN_IMAGE_SIZE,
            flatten=False,
            use_layer_norm=False):
        super().__init__()
        h, w = image_size

        if type(kernels) is list:
            if len(kernels) == 1:
                kernel_size = kernels[0]
            else:
                kernel_size = tuple(kernels)
        else:
            kernel_size = kernels

        if repeats is None:
            repeats = [1 for i in range(len(hidden_dims))]
        else:
            assert len(repeats) == len(hidden_dims)

        conv_blocks = []
        current_dim = input_dim
        # NOTE: The last hidden dim is the output dim
        for rep, hidden_dim in zip(repeats, hidden_dims):
            block = ConvBlock(current_dim, hidden_dim, h, w,
                repeats=rep,
                kernel_size=kernel_size,
                residual_link=residual_link,
                use_layer_norm=use_layer_norm)
            current_dim = hidden_dim
            conv_blocks.append(block)
            h, w = block.output_size

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.flatten = flatten
        self.output_dim = hidden_dims[-1]
        self.output_image_size = (h, w)
        # self.output_size = hidden_dims[-1] * h * w

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # default: image_size = (80, 80)
        # batch, input_dim, 80, 80
        # batch, hidden_dim[0], 40, 40
        # batch, hidden_dim[1], 20, 20
        # batch, hidden_dim[2], 10, 10
        # batch, hidden_dim[3], 5, 5
        if self.flatten:
            x = x.flatten(1, -1)
            # batch, hidden_dim[4] * 5 * 5
        return x


# adapt from https://pytorch.org/docs/master/_modules/torchvision/models/resnet.html
class ResNet(nn.Module):
    def __init__(self,
            block,
            repeats,
            inplanes=64,
            channels=[64, 128, 256, 512],
            input_dim=3,
            zero_init_residual=False,
            norm_layer=None,
            enable_maxpool=True):
        super(ResNet, self).__init__()

        assert repeats is not None
        nr_layers = len(repeats)
        assert len(channels) == nr_layers

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(input_dim, self.inplanes,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = None
        if enable_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(nr_layers):
            stride = 2 if i > 0 else 1
            layers.append(self._make_layer(
                block, channels[i], repeats[i], stride=stride))
        self.layers = nn.Sequential(*layers)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layers(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetWrapper(nn.Module):
    def __init__(self, repeats=[2, 2, 2], inplanes=8, channels=[8, 16, 32],
            input_dim=1, image_size=ORIGIN_IMAGE_SIZE):
        super().__init__()
        self.resnet = ResNet(
            block=BasicBlock,
            repeats=repeats,
            inplanes=inplanes,
            channels=channels,
            input_dim=input_dim,
            enable_maxpool=False)
        h, w = image_size
        for i in range(len(repeats)):
            h = h // 2
            w = w // 2
        self.output_dim = channels[-1]
        self.output_image_size = (h, w)

    def forward(self, x):
        # input.shape: 1, h, w
        # after conv1: inplanes, h//2, w//2
        # after layer1: inplanes, h//2, w//2
        # after layer2: inplanes, h//4, w//4
        # after layer3: inplanes, h//8, w//8
        # after layer2: inplanes, h//16, w//16
        return self.resnet(x)
