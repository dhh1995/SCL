#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : model.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/21/2019
#
# Distributed under terms of the MIT license.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacinle.logging import get_logger
from jactorch.quickstart.models import MLPModel

from .baselines import SimpleModel, SharedModel
from .symbolic_model import AnalogyModel
from .utils import compute_entropy, compute_mi
from .visual_model import Visual2Symbolic
from ..constant import ORIGIN_IMAGE_SIZE, MAX_VALUE

logger = get_logger(__file__)

__all__ = ['Model']


"""
    The model is composed of two parts, one from visual to symbolic, another
    from symbolic to prediction. The losses are computed for both symbolic
    format and the final prediction, so it can fit all three kind of tasks:
    from visual to symbolic, from visual to predicion, and from symbolic to
    prediction.

    Use prediction_beta (default 1.0) and symbolic_beta (default 0.0)
    to control the loss.
"""
class Model(nn.Module):
    def __init__(self,
            nr_features,
            model_name='analogy',
            nr_experts=5,
            shared_group_mlp=True,
            one_hot=False,
            v2s_softmax=False,
            max_value=MAX_VALUE,
            not_use_softmax=False,
            visual_inputs=False,
            factor_groups=1,
            split_channel=False,
            image_size=ORIGIN_IMAGE_SIZE,
            use_layer_norm=False,
            use_resnet=False,
            conv_hidden_dims=[],
            conv_repeats=None,
            conv_kernels=3,
            conv_residual_link=True,
            nr_visual_experts=1,
            visual_mlp_hidden_dims=[],
            transformed_spatial_dim=None,
            mlp_transform_hidden_dims=[],
            exclude_angle_attr=False,
            prediction_beta=1.0,
            symbolic_beta=0.0,
            embedding_dim=None,
            embedding_hidden_dims=[],
            enable_residual_block=False,
            use_ordinary_mlp=False,
            enable_rb_after_experts=False,
            feature_embedding_dim=1,
            hidden_dims=[16],
            reduction_groups=[3],
            sum_as_reduction=0,
            lastmlp_hidden_dims=[],
            nr_context=8,
            nr_candidates=8,
            collect_inter_key=None):
        super().__init__()
        # nr_features means the num of features for the symbolic representation.
        # When it is the visual inputs case, nr_feature can also be the
        # output dimension of the visual model. If this dimension matches the
        # symbolic representation, the symbolic_loss is enabled.
        # In one-hot case, nr_feature will be multiplied by max_value.
        # symbolic loss can be weighted by symbolic_beta, as aux_loss.
        self.original_nr_feature = nr_features
        if one_hot:
            logger.warning(('The nr_feature has been multiplied by {} to fit '
                'the one hot representation. before: {}, after: {}').format(
                max_value, nr_features, nr_features * max_value))
            nr_features *= max_value

        self.visual_inputs = visual_inputs
        if visual_inputs:
            self.v2s = Visual2Symbolic(
                input_dim=1,
                shared_group_mlp=shared_group_mlp,
                conv_hidden_dims=conv_hidden_dims,
                output_dim=nr_features,
                use_resnet=use_resnet,
                conv_repeats=conv_repeats,
                conv_kernels=conv_kernels,
                conv_residual_link=conv_residual_link,
                nr_visual_experts=nr_visual_experts,
                mlp_hidden_dims=visual_mlp_hidden_dims,
                groups=factor_groups,
                split_channel=split_channel,
                transformed_spatial_dim=transformed_spatial_dim,
                mlp_transform_hidden_dims=mlp_transform_hidden_dims,
                image_size=image_size,
                use_layer_norm=use_layer_norm)
            self.symbolic_loss = nn.MSELoss()
            if one_hot:
                self.symbolic_loss = nn.CrossEntropyLoss()

        def get_symbolic_model(name):
            assert name in ['analogy', 'simple', 'shared'], \
                'Unknown model name: {}'.format(name)
            if name == 'analogy':
                return AnalogyModel(
                    nr_features=nr_features,
                    nr_experts=nr_experts,
                    shared_group_mlp=shared_group_mlp,
                    not_use_softmax=not_use_softmax,
                    embedding_dim=embedding_dim,
                    embedding_hidden_dims=embedding_hidden_dims,
                    enable_residual_block=enable_residual_block,
                    enable_rb_after_experts=enable_rb_after_experts,
                    feature_embedding_dim=feature_embedding_dim,
                    hidden_dims=hidden_dims,
                    reduction_groups=reduction_groups,
                    sum_as_reduction=sum_as_reduction,
                    lastmlp_hidden_dims=lastmlp_hidden_dims,
                    use_ordinary_mlp=use_ordinary_mlp,
                    nr_context=nr_context,
                    nr_candidates=nr_candidates,
                    collect_inter_key=collect_inter_key)
            SymbolicModel = SimpleModel if name == 'simple' else SharedModel
            return SymbolicModel(
                nr_features=nr_features,
                hidden_dims=hidden_dims,
                nr_context=nr_context,
                nr_candidates=nr_candidates)

        self.nr_features = nr_features
        self.one_hot = one_hot
        self.v2s_softmax = v2s_softmax
        self.max_value = max_value
        self.symbolic_model = get_symbolic_model(model_name)
        self.nr_ind_features = nr_features
        if model_name == 'analogy':
            self.nr_ind_features = self.symbolic_model.nr_ind_features
        self.collect_inter_key = collect_inter_key

        # self.nr_experts = nr_experts
        # self.not_use_softmax = not_use_softmax
        # self.embedding_dim = embedding_dim
        # self.embedding_hidden_dims = embedding_hidden_dims
        # self.enable_residual_block = enable_residual_block
        # self.hidden_dims = hidden_dims
        # self.reduction_groups = reduction_groups
        # self.sum_as_reduction = sum_as_reduction
        # self.lastmlp_hidden_dims = lastmlp_hidden_dims
        self.exclude_angle_attr = exclude_angle_attr
        self.prediction_beta = prediction_beta
        self.symbolic_beta = symbolic_beta
        self.pred_loss = nn.CrossEntropyLoss()
        self.have_shown_mismatch_warning = False

        # TODO: get name by args
        # def get_name(short=False):
        #     names = ['model']
        #     if self.visual_inputs:
        #         names.append(self.v2s.get_model_name(short))
        #     names.append('nf{}'.format(nr_features))
        #     if one_hot:
        #         names.append('oh')
        #     if prediction_beta != 1.0:
        #         names.append('pb{}'.format(prediction_beta))
        #     if symbolic_beta != 0.0:
        #         names.append('sb{}'.format(symbolic_beta))
        #     if symbolic_beta != 0.0 and exclude_angle_attr:
        #         names.append('ea')
        #     names.append(self.symbolic_model.get_model_name(short))
        #     if short:
        #         return ','.join(names)
        #     return '_'.join(names)

        # self.name = get_name()
        # self.short_name = get_name(short=True)

    def compute_symbolic_loss(self, output, target):
        if self.symbolic_beta == 0.0:
            return None
        batch, nr_images, nr_parts, nr_attrs = target.size()
        if self.one_hot:
            target = target.flatten().long()
        else:
            output = output.flatten()
            target = target.flatten().float()

        if self.exclude_angle_attr:
            flag = np.zeros([batch, nr_images, nr_parts, nr_attrs])
            flag[:, :, :, 0] = 1 # the angle attr is index 0 in nr_attrs dim
            flag = flag.reshape(-1)
            ind = np.where(flag == 0)[0]
            output = output[ind]
            target = target[ind]
            # print(output.shape, target.shape)
        return self.symbolic_loss(output, target)

    def forward(self, inputs):
        symbol = inputs['symbol']

        symbolic_loss = None
        output_dict = {}
        if self.visual_inputs:
            visual = inputs['image'].float()
            pred_symbol = self.v2s(visual)
            if self.one_hot:
                pred_symbol = pred_symbol.view(-1, self.max_value)
            # Both loss are calculated using original symbolic ground truth.
            symbolic_loss = self.compute_symbolic_loss(pred_symbol, symbol)
            if self.one_hot and self.v2s_softmax:
                pred_symbol = F.softmax(pred_symbol, dim=-1)
            output_dict['pred_symbol'] = pred_symbol
            output_dict['ind_features'] = pred_symbol
            symbol = pred_symbol
        elif self.one_hot:
            symbol = inputs['symbol_onehot']
        # NOTE: the shape of symbol is not consistent for different setup
        # resize the inputs in the symbolic model
        monitors = {}
        output = self.symbolic_model(symbol)

        trans_keys = ['ind_features', self.collect_inter_key]
        for k in trans_keys:
            if k is not None and k in output:
                output_dict[k] = output[k]
        logits = output['logits']
        label = inputs['label']
        pred = logits.argmax(dim=-1)
        # print(pred.shape)
        pred_result = pred.eq(label).float()
        output_dict['pred_result'] = pred_result
        monitors['acc'] = pred_result.mean()

        pred_loss = self.pred_loss(logits, label)
        output_dict['logits'] = logits
        output_dict['pred_loss'] = pred_loss
        if symbolic_loss:
            monitors['symbolic_loss'] = symbolic_loss
            loss = pred_loss * self.prediction_beta + \
                    symbolic_loss * self.symbolic_beta
        else:
            loss = pred_loss
        if 'latent_logits' in output:
            latent_logits = output['latent_logits']
            output_dict['latent_logits'] = latent_logits
            entropy = compute_entropy(latent_logits)
            # print(label, latent_logits.shape)
            batch_size = label.size(0)
            filtered = latent_logits[np.arange(batch_size), label]
            mi = compute_mi(filtered)
            mi = mi.mean()
            monitors['entropy'] = entropy
            monitors['mi'] = mi
        return loss, monitors, output_dict

    # def get_model_name(self, short=False):
    #     if short:
    #         return self.short_name
    #     return self.name
