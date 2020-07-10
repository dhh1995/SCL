#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : preprocess.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

'''
To preprocess the data in the data_dir to pkl files, for later usage.

The symbolic format: the feature dimension is splited into nr_parts * nr_attrs:
(nr_image, nr_parts, nr_attrs)

# Usage
python3 preprocess.py -d $DATASET_DIR -n $NUM [-v]

[NOTE] the truth number of examples being preprocessed is $NUM * 10
[NOTE] use -v to take visual inputs
'''

import argparse
import collections
import functools
import numpy as np
import os.path as osp
import pickle
import xml.etree.ElementTree as ET


DATA_SPLIT = {
    'train': [0, 1, 2, 3, 4, 5],
    'val': [6, 7],
    'test': [8, 9],
}

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', '-d', type=str, required=True,
    help='the dataset dir')
parser.add_argument('--task', '-t', nargs='+', type=str, required=True,
    choices=['center_single', 'up_down', 'left_right', 'in_out', 'in_distri',
             'distribute_four', 'distribute_nine'], help='the task')
parser.add_argument('--n', '-n', type=int, required=True,
    help='the size of the dataset div by 10')
parser.add_argument('--use-visual-inputs', '-v', action='store_true',
    help='Use visual inputs if True')
parser.add_argument('--no-symbolic-representation', '-nsr', action='store_true',
    help='do not process symbolic representation if True')
args = parser.parse_args()


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


def get_dataset_name(task):
    name = task
    if task == 'up_down':
        name = 'up_center_single_down_center_single'
    if task == 'left_right':
        name = 'left_center_single_right_center_single'
    if task == 'in_out':
        name = 'in_center_single_out_center_single'
    if task == 'in_distri':
        name = 'in_distribute_four_out_center_single'
    return name


def load_data(task, data_dir, n, mode='train'):
    dataset = []
    for i in range(n):
        for j in DATA_SPLIT[mode]:
            ind = i * 10 + j
            file_prefix = 'RAVEN_{}_{}'.format(ind, mode)
            npz_file = osp.join(data_dir, file_prefix + '.npz')
            npz_data = np.load(npz_file)
            data_dict = dict(
                label=int(npz_data['target']),
                meta_matrix=npz_data['meta_matrix'],
                file_prefix=file_prefix,
                index=ind,
                task=task)
            if args.use_visual_inputs:
                data_dict['image'] = npz_data['image']
            if args.no_symbolic_representation:
                data_dict['symbol'] = 0 # Nothing
            else:
                parse_func = get_parse_func(task)
                xml_file = osp.join(data_dir, file_prefix + '.xml')
                tree = ET.parse(xml_file)
                xml_root = tree.getroot()
                data_dict['symbol'] = parse_func(xml_root)
            dataset.append(data_dict)
    # data_loader = JacDataLoader(MyDataset(dataset))
    return dataset


def dump_dataset(dataset, fname, data_dir, use_visual_inputs=False):
    if use_visual_inputs:
        fname += '_visual'
    file_name = osp.join(data_dir, '{}.pkl'.format(fname))
    with open(file_name, 'wb') as f:
        pickle.dump(dataset, f)


def main():
    all_data = []
    merged_dataset = collections.defaultdict(list)
    for task in args.task:
        dataset_name = get_dataset_name(task)
        data_dir = osp.join(args.data_dir, dataset_name)
        for mode in ['train', 'val', 'test']:
            dataset = load_data(task, data_dir, args.n, mode)
            merged_dataset[mode].extend(dataset)
            all_data.extend(dataset)

    if len(args.task) > 1:
        # dump on the parent dir
        data_dir = args.data_dir
        dump_filename_extra = '_joint_' + '_'.join(args.task)
    else:
        # data_dir remains unchanged, dump to corresponding dir
        dump_filename_extra = ''

    for mode in ['train', 'val', 'test']:
        dump_dataset(merged_dataset[mode], fname=mode + dump_filename_extra,
            data_dir=data_dir, use_visual_inputs=args.use_visual_inputs)

    dump_dataset(all_data, fname='all' + dump_filename_extra,
        data_dir=data_dir, use_visual_inputs=args.use_visual_inputs)

if __name__ == '__main__':
    main()
