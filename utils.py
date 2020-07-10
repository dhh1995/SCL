#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

import matplotlib.pyplot as plt

from scripts.utils import plot

__all__ = ['plot_curve', 'get_exp_name', 'get_image_title']


def plot_curve(all_meters, image_title, figure_file):
    train_acc = []
    test_acc = []
    for item in all_meters:
        train_meters, test_meters = item
        train_acc.append(train_meters['acc'].avg)
        test_acc.append(test_meters['acc'].avg)
    data_dict = dict(train_acc=train_acc, test_acc=test_acc)
    plot(data_dict, image_title, figure_file)


def get_task_abbv(task):
    names = []
    if len(task) > 1:
        names.append('joint')
    for t in task:
        name = t[:2]
        if t == 'in_distri':
            name += '4'
        if t == 'distribute_four':
            name += '4'
        if t == 'distribute_nine':
            name += '9'
        names.append(name)
    return '_'.join(names)


def get_name(prefix, arr, seperator='_'):
    name = ''
    for i, h in enumerate(arr):
        name += seperator
        if i == 0:
            name += '{}'.format(prefix)
        name += '{}'.format(h)
    return name


# get the exp name by (most frequently used) args
def get_exp_name(args):
    lr_str = '{}'.format(args.lr).replace('.', '')
    name = '{}_lr{}_nf{}_bs{}_ed{}_ne{}'.format(
        get_task_abbv(args.task), lr_str, args.num_features,
        args.batch_size, args.embedding_dim, args.num_experts)
    if args.v2s_lr is not None:
        name += '_vlr{}'.format(args.v2s_lr)
    if args.lr_anneal_start is not None:
        name += '_lr_s{}_i{}_r{}'.format(
            args.lr_anneal_start, args.lr_anneal_interval, args.lr_anneal_ratio)
    if args.observe_interval is not None:
        if args.obs_lr is not None:
            name += '_olr{}'.format(args.obs_lr)
    if args.normal_group_mlp:
        name += '_ngm'
    name += get_name('hd', args.hidden_dims)
    if args.feature_embedding_dim != 1:
        name += '_fed{}'.format(args.feature_embedding_dim)
    if args.use_visual_inputs:
        if args.prediction_beta != 1.0:
            name += '_pb{}'.format(args.prediction_beta)
        if args.symbolic_beta != 0.0:
            name += '_sb{}'.format(args.symbolic_beta)
        if args.image_size != [80, 80]:
            h, w = args.image_size
            name += '_is{}_{}'.format(h, w)
        if args.use_resnet:
            name += '_ur'
        if args.num_visual_experts > 1:
            name += '_nve{}'.format(args.num_visual_experts)
        if args.factor_groups > 1:
            name += '_fg{}'.format(args.factor_groups)
            if args.split_channel:
                name += '_sc'
        if args.transformed_spatial_dim is not None:
            name += '_tsd{}'.format(args.transformed_spatial_dim)
        if args.v2s_softmax:
            name += '_vss'
        name += get_name('chd', args.conv_hidden_dims)
        if args.conv_repeats is not None:
            name += get_name('cr', args.conv_repeats)
        if args.conv_residual_link:
            name += '_crl'
        name += get_name('vhd', args.visual_mlp_hidden_dims)
        name += get_name('thd', args.mlp_transform_hidden_dims)
    name += get_name('ehd', args.embedding_hidden_dims)
    name += get_name('lhd', args.lastmlp_hidden_dims)
    name += get_name('rg', args.reduction_groups)

    if args.sum_as_reduction > 0:
        name += '_sum{}'.format(args.sum_as_reduction)
    if args.one_hot:
        name += '_oh'
    if args.enable_residual_block:
        name += '_res'
    if args.weight_decay != 0.0:
        wd_str = '{}'.format(args.weight_decay).replace('.', '')
        name += '_wd{}'.format(wd_str)
    if args.use_layer_norm:
        name += '_ln'
    if args.adjust_size:
        name += '_adj'
    if args.not_use_softmax:
        name += '_ns'
    if args.exclude_angle_attr:
        name += '_ea'
    name += args.split
    if args.random_seed is not None:
        name += '_seed{}'.format(args.random_seed)
    if args.numpy_random_seed is not None:
        name += '_nseed{}'.format(args.numpy_random_seed)
    if args.torch_random_seed is not None:
        name += '_tseed{}'.format(args.torch_random_seed)
    if len(args.extra) > 0:
        name += '_' + args.extra
    return name


def get_image_title(args):
    title = args.image_title
    if title is None:
        title = '{},lr={},nf={},ed={},ne={},hd={},rg={}'.format(
            get_task_abbv(args.task), args.lr, args.num_features,
            args.embedding_dim, args.num_experts,
            args.hidden_dims, args.reduction_groups)
        if args.v2s_lr is not None:
            title += ',vlr{}'.format(args.v2s_lr)
        if args.weight_decay != 0.0:
            title += ',wd{}'.format(args.weight_decay)
        if args.lr_anneal_start is not None:
            title += ',lr s{}_r{}'.format(
                args.lr_anneal_start, args.lr_anneal_ratio)
            if args.lr_anneal_interval != 1:
                title += '_i{}'.format(args.lr_anneal_interval)
        if args.normal_group_mlp:
            title += ',ngm'
        if args.feature_embedding_dim != 1:
            title += ',fed{}'.format(args.feature_embedding_dim)
        if args.dataset_size is not None:
            title += ',ds{}'.format(args.dataset_size)
        if args.sum_as_reduction > 0:
            title += ',sum{}'.format(args.sum_as_reduction)
        if args.use_visual_inputs:
            if args.v2s_softmax:
                title += ',vss'
            if args.symbolic_beta != 0.0:
                title += ',sb{}'.format(args.symbolic_beta)
            if args.num_visual_experts > 1:
                title += ',nve{}'.format(args.num_visual_experts)
            if args.factor_groups > 1:
                title += ',fg{}'.format(args.factor_groups)
        if args.one_hot:
            title += ',oh'
        if args.enable_residual_block:
            title += ',res'
        if args.use_layer_norm:
            title += ',ln'
        if args.adjust_size:
            title += ',adj'
        if args.not_use_softmax:
            title += ',ns'
        if args.exclude_angle_attr:
            title += ',ea'
        # if args.use_visual_inputs:
        #     title += ',visual'

        split = args.split
        if len(split) > 0:
            split = ',' + split[1:]
            title += split.replace('_any', '').replace('_all', '')
    return title
