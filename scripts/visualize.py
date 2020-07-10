#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualize.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

'''
To visualize the examples as images.

# Usage
python3 visualize.py -d $DATASET_DIR [-n $NUM | -vl $VIS_LIST_FILE] -w $NUM_WORKER
'''

import argparse
import functools
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import os.path as osp
import pickle
import time
import tqdm
import xml.etree.ElementTree as ET


# from IPython import embed

DATA_SPLIT = {
    'train': [0, 1, 2, 3, 4, 5],
    'val': [6, 7],
    'test': [8, 9],
}

parser = argparse.ArgumentParser()
parser.add_argument('--workers', '-w', type=int, default=5, 
    help='The nubmer of workers for multiprocessing')
parser.add_argument('--data-dir', '-d', type=str, required=True, 
    help='the dataset dir')
parser.add_argument('--dump-dir', '-du', type=str, default=None,
    help='the dump dir')
parser.add_argument('--vis-list', '-vl', type=str, default=None,
    help='the file contains the list of names to be vis')
parser.add_argument('--n', '-n', type=int, default=None,
    help='the size of the dataset div by 10')
# parser.add_argument('--task', '-t', type=str, required=True, 
#     choices=['center_single', 'up_down', 'left_right', 'in_out',
#              'distribute_four', 'distribute_nine'], help='the task')
args = parser.parse_args()
if args.dump_dir is None:
    args.dump_dir = args.data_dir
if not osp.exists(args.dump_dir):
    os.makedirs(args.dump_dir)

def vis(file_prefix):
    t1 = time.time()

    # xml_file = osp.join(args.data_dir, file_prefix + '.xml')
    # tree = ET.parse(xml_file)
    # xml_root = tree.getroot()
    npz_file = osp.join(args.data_dir, file_prefix + '.npz')
    dump_file = osp.join(args.dump_dir, file_prefix + '.png')
    npz_data = np.load(npz_file)

    load_time = time.time() - t1
    # ------------------------------
    t1 = time.time()

    image = npz_data['image']
    label = npz_data['target']

    fig, axes = plt.subplots(5, 4)
    axes[2, 3].text(0.2, 0.4, 'Answer: {}'.format(chr(65+label)))
    for i in range(8):
        x = i // 3
        y = i % 3
        axes[x, y].imshow(image[i], cmap='gray')
        x = i // 4
        y = i % 4
        axes[x + 3, y].imshow(image[8 + i], cmap='gray')

    plt_time = time.time() - t1
    # ------------------------------
    t1 = time.time()

    plt.savefig(dump_file)

    save_time = time.time() - t1

    return load_time, plt_time, save_time


def main():
    pool = Pool(args.workers)
    file_prefixs = []
    vis_list = args.vis_list
    if vis_list is not None:
        if vis_list.endswith('.pkl'):
            # index file
            with open(vis_list, 'rb') as f:
                inds = pickle.load(f)
            for i in inds:
                if i.endswith('.npz'):
                    file_prefixs.append(i[:-4])
        elif vis_list.endswith('.json'):
            # fail cases
            with open(vis_list, 'r') as f:
                line = f.readline()
                fail_cases = json.loads(line)
                for i in fail_cases['fail_cases']:
                    file_prefixs.append(i['file_prefix'])
        else:
            assert False, 'Unknown vis list file type, {}'.format(vis_list)
    else:
        assert args.n is not None, 'The number of examples should be provided'
        for mode in ['train', 'val', 'test']:
            for i in range(args.n):
                for j in DATA_SPLIT[mode]:
                    ind = i * 10 + j
                    file_prefixs.append('RAVEN_{}_{}'.format(ind, mode))

    nr_examples = len(file_prefixs)
    print('[INFO] {} files to be visualized.'.format(nr_examples))
    with tqdm.tqdm(total=nr_examples) as pbar:
        for t in pool.imap_unordered(vis, file_prefixs):
            message = 'load time {:.4f}, plt time {:.4f}, save time {:.4f}'.format(
                t[0], t[1], t[2])
            pbar.set_description(message)
            pbar.update()


if __name__ == '__main__':
    main()
