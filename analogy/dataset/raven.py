#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : raven.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

import copy
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET

from torch.utils.data.dataset import Dataset

from .utils import resize_images, get_parse_func
from ..constant import ORIGIN_IMAGE_SIZE, MAX_VALUE

__all__ = ['RAVENDataset']


# TODO: data will be replaced with index_file + data_dir
class RAVENDataset(Dataset):
    def __init__(self, data, 
            task=None,
            inds=None,
            use_visual_inputs=False,
            image_size=ORIGIN_IMAGE_SIZE,
            one_hot=False,
            max_value=MAX_VALUE,
            adjust_size=False):
        self.data = data
        if task is not None:
            if len(task) > 1:
                task = None
                # not supported yet
            else:
                task = task[0]
        self.task = task
        self.inds = inds
        self.length = self._get_length()
        self.use_visual_inputs = use_visual_inputs
        self.image_size = image_size
        self.one_hot = one_hot
        self.max_value = max_value
        self.adjust_size = adjust_size

    def _get_length(self):
        if self.data is not None:
            return len(self.data)
        else:
            return len(self.inds)

    def __getitem__(self, item):
        source = 'data'
        if self.data is None:
            meta_matrix_key_name = 'meta_matrix'
            # load using inds
            source = 'inds'
            file_name = self.inds[item]
            suffix = file_name.split('.')[-1]
            suffix_len = len(suffix) + 1
            file_prefix = osp.basename(file_name)[:-suffix_len]
            npz_data = np.load(file_name)
            try:
                parse_func = get_parse_func(self.task)
                xml_file = file_name[:-suffix_len] + '.xml'
                tree = ET.parse(xml_file)
                xml_root = tree.getroot()
                symbol = parse_func(xml_root)
            except Exception as e:
                symbol = 0 # in_distri
            data = dict(
                image=npz_data['image'],
                symbol=symbol,
                label=int(npz_data['target']),
                meta_matrix=npz_data[meta_matrix_key_name],
                file_prefix=file_prefix)
        else:
            data = copy.copy(self.data[item])
        if self.use_visual_inputs:
            h, w = self.image_size
            img = data['image']
            if (h, w) != ORIGIN_IMAGE_SIZE:
                img = resize_images(img, (w, h))
            data['image'] = img

        symbol = data['symbol']
        if self.adjust_size:
            copied = copy.copy(symbol)
            # 2 is the index of size, to align the arithmetic manually
            copied[:, :, 2] += 1
            symbol = copied

        if self.one_hot:
            nr_images, nr_parts, nr_attrs = symbol.shape
            total = nr_images * nr_parts * nr_attrs
            index = np.reshape(symbol, [total])
            one_hot = np.zeros([total, self.max_value], dtype='float32')
            prefix = 0
            for i in range(nr_attrs):
                one_hot[np.arange(total), index.astype('int32')] = 1
            one_hot = np.reshape(one_hot, [nr_images, nr_parts, nr_attrs, -1])
            # print(symbol.shape)
            data['symbol_onehot'] = one_hot

        data['symbol'] = symbol
        return data

    def __len__(self):
        return self.length