#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/14/2019
#
# Distributed under terms of the MIT license.

import matplotlib.pyplot as plt
import numpy as np
import json

# from IPython import embed

__all__ = [
    'get_stats',
    'get_mean_std_str',
    'plot',
    'get_best_epoch',
    'sort_by_index',
    'get_rule_pairs_from_meta_matrix',
    'show_heatmap']


def get_stats(data):
    data = np.array(data)
    if len(data) == 0:
        data = np.array([0])
        print('[Warning] getting stats for empty data')
    return dict(
        mean=data.mean(), std=data.std(), max=data.max(), min=data.min())


def get_mean_std_str(data, suffix='\t'):
    return '{:.2f}:{:.2f}{}'.format(data.mean(), data.std(), suffix)


def plot(data_dict, image_title, figure_file,
        xlabel='epochs', ylabel='accuracy'):
    legends = []
    for k in data_dict.keys():
        legends.append(k)
        data = data_dict[k]
        if type(data) is tuple:
            x, y = data
        else:
            x, y = np.arange(1, len(data) + 1), data
        plt.plot(x, y)

    plt.legend(legends)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.title(image_title)
    plt.savefig(figure_file)


def get_best_epoch(summary_file,
        upper=None, do_plot=False,
        title='Title', save_file='tmp.png',
        check_single_run=True):
    best_epoch = None
    best_test_acc = 0.0
    best_train_acc = 0.0
    train_acc = 0.0
    current_epoch = -1
    best_obs_err = None
    train_accs = []
    test_accs = []
    # obs_train_errs = []
    obs_errs = []
    obs_epochs = []
    with open(summary_file, 'r') as f:
        for line in f:
            summary = json.loads(line)
            mode = summary['mode']
            epoch = summary['epoch']
            if check_single_run:
                if upper is not None and epoch > upper:
                    break
            if mode == 'train':
                if check_single_run:
                    assert epoch > current_epoch, \
                        'summary file broken, maybe contains multiple runs'
                current_epoch = epoch
                train_acc = summary['acc']
                train_accs.append(train_acc)
            elif mode == 'val':
                if check_single_run:
                    assert current_epoch == epoch, \
                        'summary file broken, incorrect train/test epoch'
                else:
                    current_epoch = epoch
                test_acc = summary['acc']
                test_accs.append(test_acc)
                if best_epoch is None or test_acc > best_test_acc:
                    best_epoch = current_epoch
                    best_test_acc = test_acc
                    best_train_acc = train_acc
            elif mode == 'test':
                pass
            else:
                assert mode == 'obs'
                if summary['dataset_name'] == 'val':
                    obs_err = summary['stat']
                    obs_epochs.append(epoch)
                    obs_errs.append(obs_err)
                    if best_obs_err is None or obs_err < best_obs_err:
                        best_obs_err = obs_err

    message = 'Best epoch: epoch={}, train_acc={:.8f}, val_acc={:.8f}'.format(
        best_epoch, best_train_acc, best_test_acc)
    if best_obs_err is not None:
        message += ', best_obs_err={}'.format(best_obs_err)
    if do_plot:
        data_dict = dict(train_acc=train_accs, test_acc=test_accs)
        if len(obs_epochs) > 0:
            obs_epochs = np.array(obs_epochs)
            obs_errs = np.array(obs_errs)
            obs_errs = np.minimum(obs_errs, 1.0)
            data_dict['obs_err']= (obs_epochs, obs_errs)
        plot(data_dict, title, save_file)
    extra = {}
    if best_obs_err is not None:
        extra['obs_err'] = best_obs_err
    return best_epoch, best_train_acc, best_test_acc, message, extra


def sort_by_index(all_data):
    def take_index(data_dict):
        return data_dict['index']
    all_data.sort(key=take_index)
    return all_data


def get_rule_pairs_from_meta_matrix(meta_matrix):
    nr_rules = len(meta_matrix)
    nr_entries = len(meta_matrix[0])
    rule_pairs = []
    for i in range(nr_rules):
        if meta_matrix[i].sum() > 0:
            actives = []
            for j in range(nr_entries):
                if meta_matrix[i][j] > 0:
                    actives.append(j)
            if len(actives) > 2:
                assert len(actives) == 3 and actives[1:] == [4, 5], 'only (rel, NUM & POS) should exists'
            relation = actives[0]
            for j in actives[1:]:
                attr = j - 4
                rule_pairs.append((relation, attr))
    return rule_pairs


def show_heatmap(bins, dump_file):
    # ["Constant", "Progression", "Arithmetic", "Distribute_Three", "Number", "Position", "Type", "Size", "Color"]
    rows = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]
    cols = ["Number", "Position", "Type", "Size", "Color"]

    fig, ax = plt.subplots()
    im = ax.imshow(bins)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(rows)):
        for j in range(len(cols)):
            text = ax.text(j, i, '{:.4f}'.format(bins[i, j]),
                           ha="center", va="center", color="w")

    ax.set_title("The histogram of the relationships in the failure cases")
    fig.tight_layout()
    # plt.show()
    plt.savefig(dump_file)
