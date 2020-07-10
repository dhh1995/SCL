#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : model.py
# Author : Honghua Dong, Tony Wu
# Email  : dhh19951@gmail.com, tonywu0206@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.logging import get_logger

from jactorch.quickstart.models import MLPModel
from jactorch.nn import Conv2d

from .modules import FCResBlock, SharedGroupMLP, Expert, Scorer
from .utils import transform

logger = get_logger(__file__)

__all__ = ['AnalogyModel']


class AnalogyModel(nn.Module):
    def __init__(self,
            nr_features,
            nr_experts=5,
            shared_group_mlp=True,
            expert_output_dim=1,
            not_use_softmax=False,
            embedding_dim=None,
            embedding_hidden_dims=[],
            enable_residual_block=False,
            enable_rb_after_experts=False,
            feature_embedding_dim=1,
            hidden_dims=[16],
            reduction_groups=[3],
            sum_as_reduction=0,
            lastmlp_hidden_dims=[],
            use_ordinary_mlp=False,
            nr_context=8,
            nr_candidates=8,
            collect_inter_key=None):
        super().__init__()
        self.nr_context = nr_context
        self.nr_candidates = nr_candidates
        self.nr_images = nr_context + nr_candidates
        self.nr_input_images = nr_context + 1
        self.nr_experts = nr_experts
        self.feature_embedding_dim = feature_embedding_dim

        self.not_use_softmax = not_use_softmax
        self.nr_features = nr_features
        self.nr_candidates = nr_candidates
        self.collect_inter_key = collect_inter_key

        current_dim = nr_features

        self.enable_residual_block = enable_residual_block
        if self.enable_residual_block:
            self.FCblock = FCResBlock(current_dim)

        self.embedding = None
        self.embedding_dim = embedding_dim
        if embedding_dim is not None:
            self.embedding = MLPModel(current_dim, embedding_dim,
                hidden_dims=embedding_hidden_dims)
            current_dim = embedding_dim

        assert feature_embedding_dim > 0
        assert current_dim % feature_embedding_dim == 0, (
            'feature embedding dim should divs current dim '
            '(nr_feature or embedding_dim)')
        current_dim = current_dim // feature_embedding_dim
        self.nr_ind_features = current_dim

        self.group_input_dim = self.nr_input_images * feature_embedding_dim
        # experts = [Expert(self.group_input_dim, hidden_dims)
        #     for i in range(nr_experts)]
        # self.experts = nn.ModuleList(experts)
        assert expert_output_dim == 1, 'only supports expert_output_dim == 1'

        groups = current_dim
        # group_size = feature_embedding_dim
        group_output_dim = expert_output_dim
        if use_ordinary_mlp:
            groups = 1
            # group_size = current_dim * feature_embedding_dim
            self.group_input_dim *= current_dim
            group_output_dim = self.nr_ind_features * expert_output_dim

        self.experts = SharedGroupMLP(
            groups=groups,
            group_input_dim=self.group_input_dim,
            group_output_dim=group_output_dim,
            hidden_dims=hidden_dims,
            add_res_block=enable_rb_after_experts,
            nr_mlps=nr_experts,
            shared=shared_group_mlp)

        self.scorer = Scorer(current_dim, nr_experts,
            hidden_dims=lastmlp_hidden_dims,
            reduction_groups=reduction_groups,
            sum_as_reduction=sum_as_reduction)

    def forward(self, x):
        current_dim = self.nr_features
        x = x.view(-1, self.nr_images, current_dim).float()
        # x.shape: (batch, nr_images, current_dim)

        if self.enable_residual_block:
            x = self.FCblock(x)

        if self.embedding:
            x = x.view(-1, current_dim)
            # x.shape: (batch * nr_images, current_dim)
            x = self.embedding(x)
            # x.shape: (batch * nr_images, embedding_dims)
            current_dim = self.embedding_dim

        x = x.view(-1, self.nr_images, current_dim)
        # x.shape: (batch, nr_images, current_dim)
        fe_dim = self.feature_embedding_dim

        ind_features = x
        # The $x$ is the extracted features from inputs
        # And the scorers regard $x$ as indenpent features
        x = transform(x,
            nr_context=self.nr_context, nr_candidates=self.nr_candidates)
        # x.shape: (batch, nr_candidates, nr_context + 1, current_dim)
        nr_ind_features = self.nr_ind_features

        # Using SharedGroupMLP
        nr_input_images = self.nr_input_images
        x = x.view(-1, nr_input_images, current_dim)
        # x.shape: (batch * nr_candidates, nr_input_images, current_dim)
        # experts: split groups over the last dim, with group size fed
        # each group corresponding to a feature

        container = None
        inter_layer = 0
        ci_key = self.collect_inter_key
        if ci_key is not None and ci_key.startswith('sgm_inter'):
            container = []
            inter_layer = int(ci_key[-1])
        latent_logits = self.experts(x,
            inter_results_container=container, inter_layer=inter_layer)
        latent_logits = latent_logits.view(-1, nr_ind_features, self.nr_experts)
        # latent_logits.shape: (batch * nr_candidates,
        #     nr_ind_features * nr_experts * expert_output_dim)

        # # Using Expert
        # x = x.view(-1, self.nr_candidates, nr_input_images, nr_ind_features, fe_dim)
        # # x.shape: (batch, nr_candidates, nr_input_images, nr_ind_features, fe_dim)
        # x = x.permute(0, 1, 3, 2, 4).contiguous()
        # # x.shape: (batch, nr_candidates, nr_ind_features, nr_input_images, fe_dim)
        # x = x.view(-1, self.group_input_dim)
        # # x.shape: (batch * nr_candidates * nr_ind_features, group_input_dim)
        # latent_logits = torch.cat([
        #     expert(x) for expert in self.experts], dim=-1)
        # # latent_logits/x.shape: (batch * nr_candidates * nr_ind_features, nr_experts)

        if self.not_use_softmax:
            x = latent_logits
        else:
            x = F.softmax(latent_logits, dim=-1)
        latent_logits = latent_logits.view(-1,
            self.nr_candidates, nr_ind_features, self.nr_experts)

        # x.shape: (batch * nr_candidates * nr_ind_features, nr_experts)
        x = x.view(-1, nr_ind_features, self.nr_experts)
        # x.shape: (batch * nr_candidates, nr_ind_features, nr_experts)
        x = self.scorer(x)

        x = x.view(-1, self.nr_candidates)
        results = dict(logits=x,
            latent_logits=latent_logits,
            ind_features=ind_features)
        if ci_key is not None and ci_key.startswith('sgm_inter'):
            sgm_inter = torch.cat(container, dim=-1)
            num = sgm_inter.size(-1)
            sgm_inter = sgm_inter.view(
                -1, self.nr_candidates, nr_ind_features, num)
            results[ci_key] = sgm_inter
        return results
