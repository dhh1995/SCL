#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Honghua Dong, Tony Wu
# Email  : dhh19951@gmail.com, tonywu0206@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

# TODO: compute mutual information instead of entropy

import torch
import torch.nn.functional as F

__all__ = ['compute_entropy', 'compute_mi', 'transform', 'vis_transform']


def compute_mi(logits, eps=1e-8):
    # logits.shape: (batch, current_dim, nr_experts)
    logits = logits.permute(1, 0, 2).contiguous()
    # logits.shape: (current_dim, batch, nr_experts)
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy = -(policy * log_policy).sum(dim=-1)
    H_expert_given_x = entropy.mean(dim=-1)

    avg_policy = policy.mean(dim=-2)
    log_avg_policy = (avg_policy + eps).log()
    H_expert = -(avg_policy * log_avg_policy).sum(dim=-1)
    return H_expert - H_expert_given_x


def compute_entropy(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy = -(policy * log_policy).sum(dim=-1)
    return entropy.mean()


'''
To fill the candidates in and regard them as batch
'''
def transform(inputs, nr_context=8, nr_candidates=8):
    context = inputs.narrow(1, 0, nr_context)
    # context.shape: (batch, nr_context, nr_features)
    candidates = inputs.narrow(1, nr_context, nr_candidates)
    # candidates.shape: (batch, nr_candidates, nr_features)
    context = context.unsqueeze(1)
    # context.shape: (batch, 1, nr_context, nr_features)
    context = context.expand(-1, nr_candidates, -1, -1)
    # context.shape: (batch, nr_candidates, nr_context, nr_features)
    candidates = candidates.unsqueeze(2)
    # candidates.shape: (batch, nr_candidates, 1, nr_features)
    merged = torch.cat([context, candidates], dim=2)
    # merged.shape: (batch, nr_candidates, nr_context + 1, nr_features)
    return merged


def vis_transform(inputs, nr_context=8, nr_candidates=8):
    context = inputs.narrow(1, 0, nr_context)
    # context.shape: (batch, nr_context, IMG_SIZE, IMG_SIZE)
    candidates = inputs.narrow(1, nr_context, nr_candidates)
    # candidates.shape: (batch, nr_candidates, IMG_SIZE, IMG_SIZE)
    context = context.unsqueeze(1)
    # context.shape: (batch, 1, nr_context, IMG_SIZE, IMG_SIZE)
    context = context.expand(-1, nr_candidates, -1, -1, -1)
    # context.shape: (batch, nr_candidates, nr_context, IMG_SIZE, IMG_SIZE)
    candidates = candidates.unsqueeze(2)
    # candidates.shape: (batch, nr_candidates, 1, IMG_SIZE, IMG_SIZE)
    merged = torch.cat([context, candidates], dim=2)
    # merged.shape: (batch, nr_candidates, nr_context + 1, IMG_SIZE, IMG_SIZE)
    return merged

