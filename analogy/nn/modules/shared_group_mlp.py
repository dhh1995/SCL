#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : shared_group_mlp.py
# Author : Honghua Dong, Tony Wu
# Email  : dhh19951@gmail.com, tonywu0206@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.logging import get_logger
from jactorch.nn import Conv1dLayer
from jactorch.quickstart.models import MLPModel

from .modules import FCResBlock

logger = get_logger(__file__)

__all__ = ['GroupMLP', 'SharedGroupMLP']


# Use conv1D with group param as group mlp
class GroupMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], groups=1,
            batch_norm=None, dropout=None, activation='relu', flatten=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert input_dim % groups == 0

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)

        layers = []
        nr_layers = len(dims) - 1
        for i in range(nr_layers):
            assert dims[i + 1] % groups == 0
            if i + 1 < nr_layers:
                layers.append(Conv1dLayer(dims[i], dims[i + 1],
                    kernel_size=1, groups=groups, batch_norm=batch_norm,
                    dropout=dropout, activation=activation))
            else: # last layer
                layers.append(Conv1dLayer(dims[i], dims[i + 1], kernel_size=1,
                    groups=groups, bias=True))
        self.mlp = nn.ModuleList(layers)
        # self.flatten = flatten

    def forward(self, x):
        shape = list(x.size())
        # x.shape: (batch, *, input_dim)
        x = x.flatten(0, -2)
        # x.shape: (batch', input_dim)
        x = x.unsqueeze(-1)
        # x.shape: (batch', input_dim, 1)
        for layer in self.mlp:
            x = layer(x)
        # x.shape: (batch', output_dim, 1)
        shape[-1] = self.output_dim
        x = x.view(*shape)
        # x.shape: (batch, *, output_dim)
        return x

"""
    SharedGroupMLP: split over the last dimension
    and put to the batch dimension, then apply MLP.
"""
class SharedGroupMLP(nn.Module):
    # shared mlp over groups splited over the last dim
    # take the last two dims as input dims (while the last one is splitted)
    def __init__(self, groups, group_input_dim, group_output_dim,
            hidden_dims=[], add_res_block=True, nr_mlps=1, flatten=True,
            shared=True):
        super().__init__()
        self.shared = shared
        self.groups = groups
        self.group_input_dim = group_input_dim
        self.group_output_dim = group_output_dim
        # mlps indicates different experts
        if shared:
            mlps = [MLPModel(group_input_dim, group_output_dim,
                hidden_dims=hidden_dims, flatten=False) for i in range(nr_mlps)]
        else:
            mlps = [GroupMLP(group_input_dim * groups, group_output_dim * groups,
                hidden_dims=hidden_dims) for i in range(nr_mlps)]
        self.mlps = nn.ModuleList(mlps)
        self.FCblocks = None
        if shared and add_res_block:
            FCblocks = [FCResBlock(group_output_dim) for i in range(nr_mlps)]
            self.FCblocks = nn.ModuleList(FCblocks)
        self.flatten = flatten

    def forward(self, x, inter_results_container=None, inter_layer=0):
        assert x.size(-1) % self.groups == 0
        group_size = x.size(-1) // self.groups
        xs = x.split(group_size, dim=-1)
        new_xs = []
        for i in xs:
            # apply on last two axis, the last axis is splitted
            x = i.flatten(-2, -1)
            # x.shape: (batch, *, group_input_dim)
            new_xs.append(x)
        x = torch.stack(new_xs, dim=-2)
        # x.shape: (batch, *, groups, group_input_dim)
        if not self.shared:
            x = x.flatten(-2, -1)
            # x.shape: (batch, *, groups * group_input_dim)
        ys = []
        for ind, mlp in enumerate(self.mlps):
            if inter_results_container is not None:
                nr_layers = min(inter_layer, len(mlp.mlp))
                t = x
                for j in range(nr_layers):
                    t = mlp.mlp[j](t)
                # t.shape: (batch, *, groups, hidden_dims[nr_layers - 1])
                inter_results_container.append(t)
            y = mlp(x)
            if self.FCblocks:
                y = self.FCblocks[ind](y)
            ys.append(y)
        x = torch.cat(ys, dim=-1)
        # [no-share] x.shape: (batch, *, nr_mlps * groups * group_output_dim)
        # [shared] x.shape: (batch, *, groups, nr_mlps * group_output_dim)
        if self.shared and self.flatten:
            x = x.flatten(-2, -1)
            # x.shape: (batch, *, groups * nr_mlps * group_output_dim)
        return x
