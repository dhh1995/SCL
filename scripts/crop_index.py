#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : crop_index.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 01/14/2020
#
# Distributed under terms of the MIT license.

'''
To filter the index file according to a unix-style regex matching.

# Usage
python3 crop_index.py -i $INPUT_FILE_NAME -o $OUTPUT_FILE_NAME -n $NUM
'''

import argparse
import os.path as osp
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', '-i', type=str, required=True, 
    help='the input file')
parser.add_argument('--output-file', '-o', type=str, required=True,
    help='the output file')
parser.add_argument('--num', '-n', type=int, required=True,
    help='the number to be remained')
parser.add_argument('--seed', '-se', type=int, default=0,
    help='the random seed')
parser.add_argument('--random-crop', '-r', action='store_true',
    help='randomly shuffle the indexes before crop')
parser.add_argument('--verbose', '-v', action='store_true',
    help='print the result')

args = parser.parse_args()
random.seed(args.seed)

def main():
    with open(args.input_file, 'rb') as f:
        index = pickle.load(f)
    if args.random_crop:
        random.shuffle(index)
    result = index[:args.num]
    print('{} files resulted after crop'.format(len(result)))
    if args.verbose:
        print(result)
    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
