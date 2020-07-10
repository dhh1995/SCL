#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : split_train_val.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

'''
To split an all.pkl (a dataset file) into {train/val}_split_{rel}_{attr}_{args}.pkl

# Usage
python3 split_train_val.py $DATASET_FILE -r $REL(s) -a $ATTR(s)

# [NOTE] '-indenp' can process all required split for a table result (only held-out a certain pair)
'''

import argparse
import numpy as np
import os.path as osp
import pickle

from utils import get_rule_pairs_from_meta_matrix

# from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('dataset_file', type=str, help='the dataset file')
# parser.add_argument('--task', '-t', type=str, required=True, 
#     choices=['center_single', 'up_down', 'left_right', 'in_out',
#              'distribute_four', 'distribute_nine'], help='the task')
parser.add_argument('--relations', '-r', type=int, nargs='+', required=True,
    help='the held-out relations for (rel, attr) pairs, 0:Const, 1:Pro, 2:Arith, 3:Union')
parser.add_argument('--attributes', '-a', type=int, nargs='+', required=True,
    help='the helo-out attributes for (rel, attr) pairs, 0:Num, 1:Pos, 2:Type, 3:Size, 4:Color')
parser.add_argument('--list-format', '-lf', action='store_true',
    help='regard the rels and attrs as list of pairs, rather the tensor prod, if True')
parser.add_argument('--all-belong-to', '-all', action='store_true',
    help='split to val when all (instead of any) rule_pairs of data belong to held-out-pairs, if True')
parser.add_argument('--dump-name', '-name', type=str, default=None,
    help='the dump name')
parser.add_argument('--use-visual-inputs', '-v', action='store_true',
    help='Use visual inputs if True')
parser.add_argument('--independent-split', '-indenp', action='store_true',
    help='regard the held-out pairs independently, and split for each of them')
parser.add_argument('--inds-only', '-io', action='store_true',
    help='only dumps inds if True')

args = parser.parse_args()

# relations are represented by a 8x9 meta matrix
# Meta matrix format
# ["Constant", "Progression", "Arithmetic", "Distribute_Three", 
#  "Number", "Position", "Type", "Size", "Color"]

# check whether this data-point should be held out
def held_out(meta_matrix, held_out_pairs, all_belong_to=False):
    flag = True
    rule_pairs = get_rule_pairs_from_meta_matrix(meta_matrix)
    for rule in rule_pairs:
        if rule in held_out_pairs:
            if not all_belong_to: # any belong to
                return True
        else:
            # not belong to, so the flag becomes false
            flag = False
    return flag


# Return two parts, held out some specific data as val_data according to the split method
def process(dataset, held_out_pairs, all_belong_to=False):
    train_split = []
    train_split_inds = []
    val_split = []
    val_split_inds = []
    for data in dataset:
        meta_matrix = data['meta_matrix']
        file_name = data['file_prefix'] + '.npz'
        if held_out(meta_matrix, held_out_pairs, all_belong_to=all_belong_to):
            val_split.append(data)
            val_split_inds.append(file_name)
        else:
            train_split.append(data)
            train_split_inds.append(file_name)
    return (train_split, train_split_inds), (val_split, val_split_inds)


def to_str(list_of_int):
    s = ''
    for i in list_of_int:
        s += str(i)
    return s


def get_dump_name(relations, attributes, all_belong_to=False, list_format=False):
    dump_name = 'split_r{}_a{}'.format(to_str(relations), to_str(attributes))
    if all_belong_to:
        dump_name += '_all'
    else:
        dump_name += '_any'
    if list_format:
        dump_name += '_lf'
    if args.use_visual_inputs:
        dump_name += '_visual'
    return dump_name


def dump_dataset(train, val, data_dir, dump_name):
    for mode, dataset in zip(['train', 'val'], [train, val]):
        data, inds = dataset
        print('{}_dataset, num:{}'.format(mode, len(data)))
        if not args.inds_only:
            file_name = osp.join(data_dir, '{}_{}.pkl'.format(mode, dump_name))
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)
            print('[data file] {} saved'.format(file_name))
        file_name = osp.join(data_dir, '{}_{}_inds.pkl'.format(mode, dump_name))
        with open(file_name, 'wb') as f:
            pickle.dump(inds, f)
        print('[index file] {} saved'.format(file_name))


def main():
    held_out_pairs = []
    if args.list_format:
        assert len(args.relations) == len(args.attributes), \
            'in the list_format, nr_rel=nr_attr should holds'
        for rel, attr in zip(args.relations, args.attributes):
            held_out_pairs.append((rel, attr))
    else:
        for rel in args.relations:
            for attr in args.attributes:
                held_out_pairs.append((rel, attr))
    # print(held_out_pairs)

    with open(args.dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    data_dir = osp.dirname(args.dataset_file)

    if args.independent_split:
        for r, a in held_out_pairs:
            dump_name = get_dump_name([r], [a])
            if args.dump_name is not None:
                dump_name = args.dump_name + dump_name
            print('the name of the dump is: {}'.format(dump_name))
            train, val = process(dataset, [(r, a)])
            dump_dataset(train, val, data_dir, dump_name)
    else:
        dump_name = args.dump_name
        if dump_name is None:
            dump_name = get_dump_name(
                args.relations, args.attributes, args.all_belong_to, args.list_format)
        print('the name of the dump is: {}'.format(dump_name))

        train, val = process(
            dataset, held_out_pairs, all_belong_to=args.all_belong_to)
        dump_dataset(train, val, data_dir, dump_name)


if __name__ == '__main__':
    main()
