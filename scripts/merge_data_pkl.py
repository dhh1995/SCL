#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : merge_data_pkl.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 02/03/2020
#
# Distributed under terms of the MIT license.

'''
To merge data pkl file into one

Usage:
python3 scripts/merge_data_pkl.py $PKLS -du $DUMP_DIR
'''

import argparse
import numpy as np
import os.path as osp
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('pkls', type=str, nargs='+', help='the metrics files (or prefix)')
parser.add_argument('--dump-file', '-du', type=str, default='merged.pkl',
    help='the dump file name for the merged result')
args = parser.parse_args()

def main():
    all_data = []
    for pkl_file in args.pkls:
        with open(pkl_file, 'rb') as f:
            all_data.extend(pickle.load(f))
        print('{} file loaded'.format(pkl_file))
    with open(args.dump_file, 'wb') as f:
        pickle.dump(all_data, f)
    print('data dumped to {}'.format(args.dump_file))


if __name__ == '__main__':
    main()
