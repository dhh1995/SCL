#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : baselines.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.logging import get_logger
from jactorch.quickstart.models import MLPModel

from .utils import transform

logger = get_logger(__file__)

__all__ = ['SimpleModel', 'SharedModel']


class SimpleModel(nn.Module):
    def __init__(self,
            nr_features,
            hidden_dims=[32, 16],
            output_dim=None,
            activation='relu',
            nr_context=8,
            nr_candidates=8):
        super().__init__()
        self.nr_images = nr_context + nr_candidates
        self.nr_features = nr_features
        self.input_dim = self.nr_images * nr_features
        if output_dim is None:
            output_dim = nr_candidates
        self.mlp = MLPModel(
            input_dim=self.input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation)

    def forward(self, x):
        x = x.view(-1, self.nr_images, self.nr_features).float()
        # x.shpae: (batch, nr_images, nr_features)
        x = x.view(-1, self.input_dim)
        # x.shpae: (batch, nr_images * nr_features)
        x = self.mlp(x)
        # x.shpae: (batch, nr_candidates)
        return dict(logits=x)


class SharedModel(nn.Module):
    def __init__(self,
            nr_features,
            hidden_dims=[32, 16],
            activation='relu',
            nr_context=8,
            nr_candidates=8):
        super().__init__()
        self.nr_context = nr_context
        self.nr_candidates = nr_candidates
        self.nr_features = nr_features
        self.nr_images = nr_context + nr_candidates
        self.input_dim = (self.nr_context + 1) * nr_features
        self.mlp = MLPModel(
            input_dim=self.input_dim,
            output_dim=1, # predict one score
            hidden_dims=hidden_dims,
            activation=activation)

    def forward(self, x):
        x = x.view(-1, self.nr_images, self.nr_features).float()
        # x.shpae: (batch, nr_images, nr_features)
        x = transform(x, nr_context=self.nr_context,
            nr_candidates=self.nr_candidates)
        # x.shape: (batch, nr_candidates, nr_context + 1, nr_features)
        x = x.view(-1, self.input_dim)
        # x.shape: (batch * nr_candidates, (nr_context + 1) * nr_features)
        x = self.mlp(x)
        # x.shape: (batch * nr_candidates, 1)
        x = x.view(-1, self.nr_candidates)
        # x.shape: (batch, nr_candidates)
        return dict(logits=x)
