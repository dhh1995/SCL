#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : prepare_index.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 01/14/2020
#
# Distributed under terms of the MIT license.

'''
To prepare the index file of the data_dir to pkl file, for later usage.

# Usage
python3 prepare_index.py -d $DATASET_DIR -o $OUTPUT_FILE_NAME -filter $FILTER
'''

import argparse
import glob
import os.path as osp
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', '-d', type=str, help='the dataset dir')
parser.add_argument('--output-file', '-o', type=str, required=True,
    help='the output file')
parser.add_argument('--filter', '-filter', type=str, default='*',
    help='the filter (default: all files)')
parser.add_argument('--verbose', '-v', action='store_true',
    help='print the result')
args = parser.parse_args()


def main():
    file_names = glob.glob(osp.join(args.data_dir, args.filter))
    print('{} files listed'.format(len(file_names)))
    base_names = list(map(lambda x: osp.basename(x), file_names))
    if args.verbose:
        print(base_names)
    with open(args.output_file, 'wb') as f:
        pickle.dump(base_names, f)


if __name__ == '__main__':
    main()
