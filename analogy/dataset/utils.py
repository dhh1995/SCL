#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

import functools
import numpy as np
import os.path as osp
import pickle
from PIL import Image

__all__ = ['get_dataset_name_and_num_features',
    'load_data', 'resize_images', 'get_parse_func']


def get_dataset_name_and_num_features(task):
    name = task
    if task == 'center_single':
        num_features = 4
    elif task == 'up_down':
        name = 'up_center_single_down_center_single'
        num_features = 8
    elif task == 'left_right':
        name = 'left_center_single_right_center_single'
        num_features = 8
    elif task == 'in_out':
        name = 'in_center_single_out_center_single'
        num_features = 8
    elif task == 'in_distri':
        name = 'in_distribute_four_out_center_single'
        num_features = 5 * 7
    elif task == 'distribute_four':
        num_features = 4 * 7
    elif task == 'distribute_nine':
        num_features = 9 * 7
    else:
        assert False

    return name, num_features


def load_data(data_dir, fname):
    file_name = osp.join(data_dir, '{}.pkl'.format(fname))
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def resize_images(imgs, size):
    resized_imgs = []
    num = imgs.shape[0]
    for ind in range(num):
        image = np.array(Image.fromarray(imgs[ind]).resize(size))
        resized_imgs.append(image)
    return np.stack(resized_imgs)


def extract_attrs(data):
    return [
        int(data['Angle']),
        int(data['Color']),
        int(data['Size']),
        int(data['Type'])]


def parse_center(xml_root):
    inputs = []
    for i in range(16):
        # print(root[0][i][0][0][0][0].attrib)
        data = extract_attrs(xml_root[0][i][0][0][0][0].attrib)
        inputs.append([data])
    symbol = np.array(inputs)
    return symbol


def parse_two_parts(xml_root):
    inputs = []
    for i in range(16):
        # print(root[0][i][0][0][0][0].attrib)
        data = []
        for p in range(2):
            data.append(extract_attrs(xml_root[0][i][0][p][0][0].attrib))
        inputs.append(data)
    symbol = np.array(inputs)
    return symbol


def get_float_arr(inp):
    n = len(inp)
    i, j = 0, 0
    data = []
    while i < n:
        while inp[i] == '[' or inp[i] == ']' or inp[i] == ' ':
            i += 1
        j = i
        while j < n and inp[j] != ',' and inp[j] != ']':
            j += 1
        if j < n:
            data.append(float(inp[i:j]))
        i = j + 1
    return data


def parse_distribute(xml_root, nr_rows=3, nr_cols=3):
    inputs = []
    for i in range(16):
        # print(root[0][i][0][0][0][0].attrib)
        panel = xml_root[0][i][0][0][0]
        nr_obj = len(panel)
        data = np.zeros([nr_rows * nr_cols, 7], dtype='int32')
        for j in range(nr_obj):
            tmp = panel[j].attrib
            bbox = get_float_arr(tmp['bbox'])
            pos = (int(bbox[0] * nr_rows), int(bbox[1] * nr_cols))
            ind = pos[0] * nr_rows + pos[1]
            info = extract_attrs(tmp)
            info.extend([pos[0], pos[1], 1])
            data[ind, :] = info
        inputs.append(data)
    symbol = np.array(inputs)
    return symbol


def parse_in_distri(xml_root, nr_rows=2, nr_cols=2):
    inputs = []
    for i in range(16):
        # print(root[0][i][0][0][0][0].attrib)
        data = np.zeros([nr_rows * nr_cols + 1, 7], dtype='int32')
        info = extract_attrs(xml_root[0][i][0][0][0][0].attrib)
        data[0, :4] = info
        # pos remains unchanged
        data[0, 6] = 1

        panel = xml_root[0][i][0][1][0]
        nr_obj = len(panel)
        for j in range(nr_obj):
            tmp = panel[j].attrib
            bbox = get_float_arr(tmp['bbox'])
            pos = (int(bbox[0] * nr_rows), int(bbox[1] * nr_cols))
            ind = pos[0] * nr_rows + pos[1] + 1
            info = extract_attrs(tmp)
            info.extend([pos[0], pos[1], 1])
            data[ind, :] = info
        # print(data)
        inputs.append(data)
    symbol = np.array(inputs)
    return symbol


def get_parse_func(name):
    if name == 'center_single':
        return parse_center
    if name == 'up_down' or name == 'left_right' or name == 'in_out':
        return parse_two_parts
    if name == 'distribute_four':
        return functools.partial(parse_distribute, nr_rows=2, nr_cols=2)
    if name == 'distribute_nine':
        return functools.partial(parse_distribute, nr_rows=3, nr_cols=3)
    if name == 'in_distri':
        return parse_in_distri
    assert False, 'Unknown task {}'.format(name)
