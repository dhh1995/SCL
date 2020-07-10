#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : play_with_summary.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

'''
To get stats from summary.json
'''

import argparse
import numpy as np
import os.path as osp
import pickle
import json

from utils import get_best_epoch, get_stats, get_mean_std_str

# from IPython import embed

NUM_RELATIONS = 4
NUM_ATTRIBUTES = 5

parser = argparse.ArgumentParser()
parser.add_argument('summary', type=str, nargs='+',
    help='the summary files (or prefix)')
parser.add_argument('--groups', '-g', type=int, nargs='+', default=None,
    help='to split the summary into groups')
parser.add_argument('--exp-name', '-exp', type=str, default='',
    help='the exp_name str to help distinguish different exps')
parser.add_argument('--runs', '-n', type=int, default=1,
    help='The number of runs to be averaged')
parser.add_argument('--relations', '-r', type=int, nargs='+',default=None,
    help='the held-out relations for (rel, attr) pairs, 0:Const, 1:Pro, 2:Arith, 3:Union')
parser.add_argument('--attributes', '-a', type=int, nargs='+', default=None,
    help='the helo-out attributes for (rel, attr) pairs, 0:Num, 1:Pos, 2:Type, 3:Size, 4:Color')
parser.add_argument('--val-acc-only', '-only', action='store_true',
    help='show the val result only if True')
parser.add_argument('--show-decimal', '-deci', action='store_true',
    help='show decimal instead of percentage')
parser.add_argument('--dump-pkl', '-du', type=str, default=None,
    help='the place to dump the stats')
parser.add_argument('--plot', '-p', action='store_true',
    help='plot the curve')
parser.add_argument('--upper', '-u', type=int, default=None,
    help='The maximum number of epochs to be considered')
parser.add_argument('--title', '-t', type=str, default='Title',
    help='The title of the image')
parser.add_argument('--save-file', '-s', type=str, default='tmp.png',
    help='The file name to be saved')

args = parser.parse_args()
if args.relations is None:
    args.relations = list(range(4))
if args.attributes is None:
    args.attributes = list(range(5))
args.include_train_acc = not args.val_acc_only


def analysis_summaries():
    nr_exps = len(args.summary)
    train_accs = np.zeros(nr_exps)
    val_accs = np.zeros(nr_exps)
    for i, summary_file in enumerate(args.summary):
        _, train_acc, val_acc, message, _ = get_best_epoch(
            summary_file,
            upper=args.upper,
            do_plot=args.plot,
            title=args.title,
            save_file=args.save_file)
        train_accs[i] = train_acc
        val_accs[i] = val_acc
        print('[{}] {}'.format(summary_file, message))
    if args.groups is not None:
        assert sum(args.groups) == nr_exps
        current_index = 0
        for g in args.groups:
            if args.include_train_acc:
                print(get_mean_std_str(
                    train_accs[current_index: current_index + g],
                    suffix=''))
            print(get_mean_std_str(
                val_accs[current_index: current_index + g], suffix=''))
            current_index += g


def main():
    analysis_summaries()


if __name__ == '__main__':
    main()
