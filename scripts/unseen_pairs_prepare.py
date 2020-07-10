#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : unseen_pairs_prepare.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 02/04/2019
#
# Distributed under terms of the MIT license.

'''
To split dataset into {train/val/test}_split_{rel}_{attr}_{args}.pkl
It will produce a set of indexes stored in pkls and can be used by specifying
both --index-file-dir and --split args of the main.py program.
[NOTE] It may require more examples to fulfill the 6k,2k,2k split regime.

# Usage
python3 unseen_pairs_prepare.py $DATA_DIR $NUM -r $REL(s) -a $ATTR(s)

# [NOTE] '-indenp' can prepare all required split for a table result 
# (only held-out a certain pair)
'''

import argparse
import collections
import numpy as np
import os
import os.path as osp
import pickle

from utils import get_rule_pairs_from_meta_matrix

# from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='the dataset file')
parser.add_argument('num', type=int, help='the dataset size')
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
parser.add_argument('--dump-dir', '-du', type=str, required=True,
    help='the dump dir for inds')
parser.add_argument('--use-visual-inputs', '-v', action='store_true',
    help='Use visual inputs if True')
parser.add_argument('--independent-split', '-indenp', action='store_true',
    help='regard the held-out pairs independently, and split for each of them')
# exclude
parser.add_argument('--exclude-relations', '-er', type=int, nargs='+', default=[],
    help='the exclude relations for (rel, attr) pairs, 0:Const, 1:Pro, 2:Arith, 3:Union')
parser.add_argument('--exclude-attributes', '-ea', type=int, nargs='+', default=[],
    help='the exclude attributes for (rel, attr) pairs, 0:Num, 1:Pos, 2:Type, 3:Size, 4:Color')
parser.add_argument('--exclude-list-format', '-elf', action='store_true',
    help='regard the ex-rels and ex-attrs as list of pairs, rather the tensor prod, if True')

args = parser.parse_args()

ORIGIN_DATA_SPLIT = {
    'train': [0, 1, 2, 3, 4, 5],
    'val': [6, 7],
    'test': [8, 9],
}

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


def dump_dataset(dump_dir, inds, name):
    for mode in ['train', 'val', 'test']:
        file_name = osp.join(dump_dir, '{}_{}_inds.pkl'.format(mode, name))
        with open(file_name, 'wb') as f:
            pickle.dump(inds[mode], f)
        print('[index file] {} saved, {} inds'.format(
            file_name, len(inds[mode])))


def process(inds, bins, exclude=[], request=[6000, 2000, 2000]):
    datasets = [collections.defaultdict(list) for i in range(len(bins))]
    all_belong_to = args.all_belong_to and not args.independent_split
    train_val_flag = False
    test_flag = False
    for i, ind in enumerate(inds):
        name = ind + '.npz'
        file_name = osp.join(args.data_dir, name)
        data = np.load(file_name)
        meta_matrix = data['meta_matrix']
        for held_out_pairs, dataset in zip(bins, datasets):
            if len(exclude) > 0 and held_out(meta_matrix, exclude):
                # exclude
                pass
            elif held_out(meta_matrix, held_out_pairs,
                    all_belong_to=all_belong_to):
                if len(dataset['test']) < request[2]:
                    dataset['test'].append(name)
                else:
                    test_flag = True
            else:
                if len(dataset['train']) < request[0]:
                    dataset['train'].append(name)
                elif len(dataset['val']) < request[1]:
                    dataset['val'].append(name)
                else:
                    train_val_flag = True
        if not args.independent_split and train_val_flag and test_flag:
            break
        if i % 1000 == 0:
            print('nr_examples', i)
            for held_out_pairs, dataset in zip(bins, datasets):
                print('held {}, train {}, val {}, test {}'.format(
                    held_out_pairs, len(dataset['train']),
                    len(dataset['val']), len(dataset['test'])))
    return datasets


def get_held_out_pairs(rels, attrs, list_format=False):
    held_out_pairs = []
    if list_format:
        assert len(rels) == len(attrs), \
            'in the list_format, nr_rel=nr_attr should holds'
        for rel, attr in zip(rels, attrs):
            held_out_pairs.append((rel, attr))
    else:
        for rel in rels:
            for attr in attrs:
                held_out_pairs.append((rel, attr))
    return held_out_pairs


def main():
    held_out_pairs = get_held_out_pairs(args.relations,
        args.attributes, args.list_format)
    exclude_pairs = get_held_out_pairs(args.exclude_relations,
        args.exclude_attributes, args.exclude_list_format)

    partition = ['' for i in range(10)]
    for k in ORIGIN_DATA_SPLIT.keys():
        for i in ORIGIN_DATA_SPLIT[k]:
            partition[i] = k

    if args.num % 10 != 0:
        print('[Warning] dataset size {} is not a multipler of 10'.format(
            args.num))
    n = args.num // 10    
    inds = []
    for i in range(n):
        for j in range(10):
            ind = i * 10 + j
            file_prefix = 'RAVEN_{}_{}'.format(ind, partition[j])
            inds.append(file_prefix)

    bins = []
    names = []
    if args.independent_split:
        for r, a in held_out_pairs:
            name = get_dump_name([r], [a])
            bins.append([(r, a)])
            names.append(name)
    else:
        name = get_dump_name(args.relations, args.attributes,
            args.all_belong_to, args.list_format)
        bins.append(held_out_pairs)
        names.append(name)

    datasets = process(inds, bins, exclude=exclude_pairs)

    print('dump_dir is {}'.format(args.dump_dir))
    os.makedirs(args.dump_dir, exist_ok=True)
    for dataset, name in zip(datasets, names):
        print('the name of the dataset is: {}'.format(name))
        dump_dataset(args.dump_dir, dataset, name)


if __name__ == '__main__':
    main()
