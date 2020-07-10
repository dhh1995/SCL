#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : filter_index.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 01/14/2020
#
# Distributed under terms of the MIT license.

'''
To filter the index file according to a unix-style regex matching.

# Usage
python3 filter_index.py -i $INPUT_FILE_NAME -o $OUTPUT_FILE_NAME -filter $FILTER
'''

import argparse
import os.path as osp
import pickle
import fnmatch

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', '-i', type=str, required=True, 
    help='the input file')
parser.add_argument('--output-file', '-o', type=str, required=True,
    help='the output file')
parser.add_argument('--filters', '-f', type=str, nargs='+', required=True,
    help='the filter')
parser.add_argument('--verbose', '-v', action='store_true',
    help='print the result')

args = parser.parse_args()


def match(name, filters):
    # The filters act as OR relationship
    for f in filters:
        if fnmatch.fnmatch(name, f):
            return True
    return False


def main():
    with open(args.input_file, 'rb') as f:
        index = pickle.load(f)
    result = []
    for name in index:
        if match(name, args.filters):
            result.append(name)
    print('{} files resulted after filter'.format(len(result)))
    if args.verbose:
        print(result)
    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
