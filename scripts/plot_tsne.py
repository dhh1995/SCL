#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot_tsne.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 02/06/2020
#
# Distributed under terms of the MIT license.

'''
To visualize the examples as images.

# Usage
python3 plot_tsne.py result_file.pkl
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, 
    help='the file contains the result to be plot')
parser.add_argument('output', type=str, 
    help='the output figure file')
parser.add_argument('--plot-id', '-id', action='store_true',
    help='plot the origin as test')
args = parser.parse_args()

def main():
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    results = data['results']
    colors = np.array(data['colors'])
    max_colors = np.max(colors) + 1
    plt.axis('off')

    if args.plot_id:
        origin = data['origin']
        x_min, x_max = np.min(results, 0), np.max(results, 0)
        results = (results - x_min) / (x_max - x_min)
        fig = plt.figure()
        for i, r in enumerate(results):
            # print(i)
            plt.text(r[0], r[1], str(origin[i]),
                color=plt.cm.Set1((colors[i] + 1)),
                fontdict={'weight': 'bold', 'size': 9})
    else:
        plots = []
        labels = ['Const', 'Progress', 'Arith', 'Union']
        # plt.rcParams['axes.prop_cycle']
        c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # x_min, x_max = np.min(results, 0), np.max(results, 0)
        # results = (results - x_min) / (x_max - x_min)
        for i in range(max_colors):
            inds = np.where(colors == i)
            array = results[inds]
            plots.append(plt.scatter(
                array[:, 0], array[:, 1], s=1, color=c[i]))

        lgnd = plt.legend(plots, labels, loc='upper left', fontsize=12)
        for i in lgnd.legendHandles:
            i._sizes = [30]

        # scatter plot
        # scatter = plt.scatter(results[:, 0], results[:, 1], s=1, c=colors)
        # plt.legend(handles=scatter.legend_elements()[0], labels=labels, 
        #     loc='upper left', fontsize=12)

    plt.savefig(args.output, format='pdf', bbox_inches='tight', pad_inches=0)
    print('figure saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
