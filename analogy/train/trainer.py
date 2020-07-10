#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trainer.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/21/2019
#
# Distributed under terms of the MIT license.

"""
Trainer, specifying the training and validation
"""

import argparse
import collections
import numpy as np
import json
import os
import os.path as osp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import jacinle.io as io

from jacinle.logging import get_logger
from jacinle.utils.meter import GroupMeters
from jacinle.utils.tqdm import tqdm_pbar
from jactorch.train import TrainerEnv
from jactorch.train.env import default_reduce_func
from jactorch.utils.meta import as_cuda, as_tensor, as_float, as_cpu

from analogy.thutils import monitor_gradrms

logger = get_logger(__file__)

__all__ = ['Trainer']


class Trainer(TrainerEnv):
    def __init__(self,
            model,
            optimizer,
            epochs,
            lr,
            use_visual_inputs=False,
            v2s_lr=None,
            lr_anneal_start=None,
            lr_anneal_interval=1,
            lr_anneal_ratio=1.0,
            obs_val_only=False,
            obs_epochs=0,
            observe_env=None,
            observe_interval=None,
            use_gpu=False,
            monitor_grads=False,
            save_interval=50,
            ckpt_dir=None,
            resume_dir=None,
            disable_resume=False,
            summary_file=None,
            fail_cases_file=None):
        super().__init__(model, optimizer)
        self.epochs = epochs
        self.lr = lr
        self.use_visual_inputs = use_visual_inputs
        if v2s_lr is None:
            v2s_lr = lr
        self.v2s_lr = v2s_lr
        self.lr_anneal_start = lr_anneal_start
        self.lr_anneal_interval = lr_anneal_interval
        assert lr_anneal_interval > 0
        self.lr_anneal_ratio = lr_anneal_ratio

        self.obs_val_only = obs_val_only
        self.obs_epochs = obs_epochs
        self.observe_env = observe_env
        self.observe_interval = observe_interval
        self.use_gpu = use_gpu
        self.monitor_grads = monitor_grads
        self.save_interval = save_interval
        self.ckpt_dir = ckpt_dir
        self.disable_resume = disable_resume
        if resume_dir is None:
            resume_dir = ckpt_dir
        self.resume_dir = resume_dir
        self.summary_file = summary_file
        self.fail_cases_file = fail_cases_file
        self.current_epoch = 0
        # resume related
        self.start_epoch = 0
        self.best_val_acc = -1e-6

    def _dump_meters(self, meters, mode, extra_dict=None):
        if self.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            meters_kv['mode'] = mode
            meters_kv['epoch'] = self.current_epoch
            if extra_dict is not None:
                meters_kv.update(extra_dict)
            with open(self.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv, compressed=True))
                f.write('\n')

    def _dump_fail_cases(self, fail_cases):
        # The fail_cases.json file can be very large, restrict to at most 200
        dump_dict = dict(epoch=self.current_epoch, fail_cases=fail_cases[:200])
        if self.fail_cases_file is not None:
            with open(self.fail_cases_file, 'a') as f:
                f.write(io.dumps_json(dump_dict, compressed=True))
                f.write('\n')

    def _resume_checkpoint(self):
        if self.disable_resume:
            return
        name = 'last_epoch.pth'
        fname = osp.join(self.resume_dir, name)
        if osp.exists(fname):
            extra = self.load_checkpoint(fname)
            self.start_epoch = extra['epoch']
            self.best_val_acc = extra['best_val_acc']

    def val_epoch(self, loader, collect_inter_key=None, extra_dict=None):
        model = self.model
        model.eval()

        meters = GroupMeters()
        fail_cases = []
        nr_iters = len(loader)

        all_inter = []
        all_gts = []
        # print(logits, label)

        with tqdm_pbar(total=nr_iters) as pbar:
            for index, data in enumerate(loader):
                file_prefix = data['file_prefix']
                subtasks = None
                if 'task' in data:
                    subtasks = data['task']
                if self.use_gpu:
                    data = as_cuda(data)
                loss, monitors, output_dict = model(data)

                loss = default_reduce_func('loss', loss)
                monitors = {
                    k: default_reduce_func(k, v) for k, v in monitors.items()}

                loss = as_float(loss)
                monitors = as_float(monitors)
                logits = output_dict['logits']
                target = data['label']
                prob = F.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1)
                pred_result = pred.eq(target)
                batch_size = target.size(0)

                # collect for tsne
                if collect_inter_key is not None:
                    inter = output_dict[collect_inter_key]
                    if collect_inter_key == 'latent_logits':
                        inter = F.softmax(inter, dim=-1)
                    meta_matrix = data['meta_matrix']
                    def get_gts(st=1, ed=4):
                        gts = meta_matrix[np.arange(batch_size), st:ed, 0:4]
                        gts = gts.max(dim=-1)[1].cpu().numpy()[:, ::-1]
                        return gts

                    gts = get_gts(1, 4)
                    if meta_matrix.size(1) > 4:
                        gts2 = get_gts(5, 8)
                        gts = np.concatenate([gts, gts2], axis=1)

                    inter = inter[np.arange(batch_size), target]
                    all_inter.append(inter.detach().cpu().numpy())
                    all_gts.append(gts)

                meters.update(monitors)
                meters.update(loss=loss, n=batch_size)
                for i in range(batch_size):
                    if subtasks is not None:
                        key_name = 'acc_{}'.format(subtasks[i])
                        meters.update({key_name: pred_result[i].item()})
                    if pred_result[i].item() == 0: # fail case
                        fail_cases.append(
                            dict(file_prefix=file_prefix[i],
                                prob=prob[i].cpu().detach().numpy().tolist(),
                                pred=pred[i].item(),
                                label=target[i].item()))
                message = 'Test: loss={:.4f}, acc={:.4f}'.format(
                    loss, meters['acc'].avg)
                pbar.set_description(message)
                pbar.update()
        logger.info(meters.format_simple('> Test  Epoch {:3d}: '.format(
            self.current_epoch), compressed=not self.monitor_grads))

        if collect_inter_key is not None:
            assert extra_dict is not None
            inter = np.concatenate(all_inter, axis=0)
            gts = np.concatenate(all_gts, axis=0)
            extra_dict['inter'] = inter
            extra_dict['gts'] = gts

        # self._dump_meters(meters, 'val')
        self._dump_fail_cases(fail_cases)
        return meters

    def train_epoch(self, loader):
        model = self.model
        optimizer = self.optimizer

        model.train()
        meters = GroupMeters()
        nr_iters = len(loader)
        import time
        t1 = time.time()
        total_time = 0
        with tqdm_pbar(total=nr_iters) as pbar:
            for index, data in enumerate(loader):

                subtasks = None
                if 'task' in data:
                    subtasks = data['task']
                if self.use_gpu:
                    data = as_cuda(data)
                t2 = time.time()
                loss, monitors, output_dict, extras = self.step(data)
                total_time += time.time() - t2
                # logits = output_dict['logits']
                # pred = logits.argmax(dim=1, keepdim=True)
                # target = data['label']
                # acc = pred.eq(target.view_as(pred)).float().mean().item()

                if subtasks is not None:
                    pred_result = output_dict['pred_result']
                    for i in range(len(subtasks)):
                        key_name = 'acc_{}'.format(subtasks[i])
                        meters.update({key_name: pred_result[i].item()})
                        # print(subtasks[i], pred_result[i].mean().item())

                batch_size = data['label'].size(0)
                if self.monitor_grads:
                    meters.update(monitor_gradrms(self.model))
                meters.update(monitors)
                meters.update(loss=loss, n=batch_size)
                meters.update(lr=self.lr)
                if self.use_visual_inputs:
                    meters.update(v2s_lr=self.v2s_lr)
                # meters.update(acc=acc, n=batch_size)

                message = 'Train: lr={:.4f}, loss={:.4f}'.format(self.lr, loss)
                message += ', acc={:.4f}'.format(meters['acc'].avg)
                if self.use_visual_inputs:
                    message += ', v2s_lr={:.4f}'.format(self.v2s_lr)
                pbar.set_description(message)
                pbar.update()
        logger.info('Total train epoch time {:.3f}, step time {:.3f}'.format(
            time.time() - t1, total_time))
        logger.info(meters.format_simple('> Train Epoch {:3d}: '.format(
            self.current_epoch), compressed=not self.monitor_grads))
        # self._dump_meters(meters, 'train')
        return meters

    # NOTE(Jan 9): need to be moved to script that in seperate runs,
    # while saving all checkpoints
    def observe(self, loader, dataset_name='train'):
        self.model.eval()
        observe_env = self.observe_env
        observe_model = observe_env.model
        observe_model.train()
        observe_model.restart()

        obs_epochs = self.obs_epochs
        if dataset_name == 'val':
            # ad-hoc fix for the val dataset
            # which has 3x less data in original split
            # Approximately use 2x instead
            obs_epochs *= 2

        for i in range(obs_epochs):
            meters = GroupMeters()
            for index, data in enumerate(loader):
                if self.use_gpu:
                    data = as_cuda(data)
                loss, monitors, output_dict = self.model(data)
                # pred_symbol = output_dict['pred_symbol']
                ind_features = output_dict['ind_features']
                label = data['symbol']
                feed_dict = dict(pred=ind_features, label=label)
                loss, monitors, obs_output_dict, extras = observe_env.step(
                    feed_dict)
                loss = as_float(loss)
                monitors = as_float(monitors)

                batch_size = label.size(0)
                meters.update(monitors)
                meters.update(loss=loss, n=batch_size)
            logger.info(meters.format_simple(
                '> Obs {} Sub Epoch {:3d}: '.format(dataset_name, i),
                    compressed=True))
        # Only dump the last sub-epoch
        extra_dict = dict(dataset_name=dataset_name)
        self._dump_meters(meters, 'obs', extra_dict=extra_dict)
        return meters, obs_output_dict

    def train_eval(self, train_loader, val_loader=None):
        self._resume_checkpoint()
        # all_meters = []
        last_epoch = self.start_epoch
        for i in range(self.start_epoch + 1, self.epochs + 1):
            self.current_epoch = i
            if self.observe_env and self.observe_interval is not None:
                if i % self.observe_interval == 0:
                    if not self.obs_val_only:
                        self.observe(train_loader, dataset_name='train')
                    if val_loader is not None:
                        self.observe(val_loader, dataset_name='val')
            if self.lr_anneal_start is not None and i >= self.lr_anneal_start:
                if (i - self.lr_anneal_start) % self.lr_anneal_interval == 0:
                    self.lr *= self.lr_anneal_ratio
                    if self.use_visual_inputs:
                        self.v2s_lr *= self.lr_anneal_ratio
                    self.decay_learning_rate(self.lr_anneal_ratio)
            train_meters = self.train_epoch(train_loader)

            val_meters = None
            if val_loader is not None:
                val_meters = self.val_epoch(val_loader)

            self._dump_meters(train_meters, 'train')
            self._dump_meters(val_meters, 'val')
            if i % self.save_interval == 0:
                self.customer_save_checkpoint(self.ckpt_dir, i)

            if val_meters is not None:
                val_acc = val_meters['acc'].avg
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    logger.info('epoch: {}, best val acc: {}'.format(i, val_acc))
                    self.customer_save_checkpoint(
                        self.ckpt_dir, i, is_best=True)

            self.customer_save_checkpoint(self.resume_dir, i, is_last=True)
            last_epoch = i
            # all_meters.append((train_meters, val_meters))
        if last_epoch > self.start_epoch:
            self.customer_save_checkpoint(self.ckpt_dir, last_epoch, is_last=True)
        # return all_meters

    def customer_save_checkpoint(self, ckpt_dir, epoch,
            is_last=False, is_best=False):
        name = 'epoch_{}.pth'.format(epoch)
        if is_last:
            name = 'last_epoch.pth'
        elif is_best:
            name = 'best_epoch.pth'
        fname = osp.join(ckpt_dir, name)
        self.save_checkpoint(fname, dict(epoch=epoch,
            best_val_acc=self.best_val_acc))
