#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : main.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/06/2019
#
# Distributed under terms of the MIT license.

'''
# Usage
jac-run main.py -d $DATASET_DIR -t $TASK
'''

import argparse
import collections
import numpy as np
import json
import os
import os.path as osp
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from jacinle.logging import get_logger, set_output_file
from jacinle.utils.printing import kvprint, kvformat
from jactorch.cli import dump_metainfo
from jactorch.data.dataloader import JacDataLoader
from jactorch.parallel import JacDataParallel
from jactorch.train import TrainerEnv

from analogy.constant import MAX_VALUE
from analogy.dataset import get_dataset_name_and_num_features, load_data
from analogy.dataset import RAVENDataset
from analogy.nn import Model, Observer
from analogy.train import Trainer
from utils import plot_curve, get_exp_name, get_image_title

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()

seeds = parser.add_argument_group('Random Seeds')
seeds.add_argument('--random-seed', '-seed', type=int, default=None,
    help='The random seed for random')
seeds.add_argument('--numpy-random-seed', '-nseed', type=int, default=None,
    help='The random seed for np.random')
seeds.add_argument('--torch-random-seed', '-tseed', type=int, default=None,
    help='The random seed for torch.random')

dataset_args = parser.add_argument_group('Dataset Related Args')
# Dataset Spec
dataset_args.add_argument('--data-dir', '-d', type=str, required=True,
    help='the dataset dir')
dataset_args.add_argument('--task', '-t', nargs='+', type=str, required=True,
    choices=['center_single', 'up_down', 'left_right', 'in_out', 'in_distri',
             'distribute_four', 'distribute_nine'], help='the task')
dataset_args.add_argument('--index-file-dir', '-ifd', type=str, default=None,
    help='the dir containing index files')
dataset_args.add_argument('--train-index-file', '-tif', type=str, default=None,
    help='the training index file defining the training dataset')
dataset_args.add_argument('--val-index-file', '-vif', type=str, default=None,
    help='the validation index file defining the validation dataset')
dataset_args.add_argument('--dataset-size', '-ds', type=int, nargs='+',
    default=None, help='the size of training/val dataset')
dataset_args.add_argument('--num-context', '-nco', type=int, default=None,
    help='the number of context images')
dataset_args.add_argument('--num-candidates', '-nca', type=int, default=None,
    help='the number of candidate images')
dataset_args.add_argument('--split', '-sp', type=str, default='',
    help='the string specify the dataset split, empty for original split')
dataset_args.add_argument('--use-test-as-val', '-tv', action='store_true',
    help='Use the test set during each validation time')
# Dataset Manupilation
dataset_args.add_argument('--trunc-train-data', '-ttd', type=int, default=None,
    help='the truncated size of training dataset (default: None)')
dataset_args.add_argument('--trunc-val-data', '-tvd', type=int, default=None,
    help='the truncated size of validation dataset (default: None)')
dataset_args.add_argument('--adjust-size', '-adj', action='store_true',
    help='adjust size attr (+1) of the inputs')
# Misc
dataset_args.add_argument('--num-workers', '-w', type=int, default=None,
    help='The number of workers for data loader')

trainer_args = parser.add_argument_group('Trainer Related Args')
# Basic
trainer_args.add_argument('--use-gpu', action='store_true',
    help='use GPU or not')
trainer_args.add_argument('--epochs', '-e', type=int, default=200,
    help='the number of epochs')
trainer_args.add_argument('--obs-epochs', '-oe', type=int, default=5,
    help='the number of sub epochs for observation stage')
trainer_args.add_argument('--batch-size', '-bs', type=int, default=128,
    help='input batch size for training (default: 128)')
trainer_args.add_argument('--eval-batch-size', '-ebs', type=int, default=32,
    help='input batch size for evaluation (default: 32)')
trainer_args.add_argument('--lr', '-lr', type=float, default=0.005,
    help='learning rate (default: 0.005)')
trainer_args.add_argument('--obs-lr', '-olr', type=float, default=None,
    help='learning rate for observation (default: None, recommend 0.2)')
trainer_args.add_argument('--observe-interval', '-oi', type=int, default=None,
    help='The interval of doing observation for feature space (default: None)')
trainer_args.add_argument('--v2s-lr', '-vlr', type=float, default=None,
    help='learning rate (default: None, the same as lr)')
trainer_args.add_argument('--lr-anneal-start', '-lrs', type=int, default=None,
    help='The epoch when learning rate start annealing (default: None)')
trainer_args.add_argument('--lr-anneal-interval', '-lri', type=int, default=1,
    help='The interval of learning rate annealing (default: 1)')
trainer_args.add_argument('--lr-anneal-ratio', '-lrr', type=float, default=1.0,
    help='learning rate annealing ratio (default: 1.0)')
trainer_args.add_argument('--weight-decay', '-wd', type=float, default=0,
    help='The weight decay factor')
trainer_args.add_argument('--obs-val-only', '-ov', action='store_true',
    help='observe the val dataset only if True')
trainer_args.add_argument('--obs-thresh', '-ot', type=float, default=0.25,
    help='The thresh used by observation module')
# Save & Load
trainer_args.add_argument('--load', '-l', type=str, default=None,
    help='load the weights from a pretrained model (default: none)')
trainer_args.add_argument('--save-interval', '-si', type=int, default=50,
    help='model save interval (epochs) (default: 50)')
# Utils
trainer_args.add_argument('--monitor-grads', '-mg', action='store_true',
    help='monitor the grad of weights if True')
trainer_args.add_argument('--test-only', '-test', action='store_true',
    help='Test only')
trainer_args.add_argument('--test-obs', '-to', action='store_true',
    help='Test obs')
trainer_args.add_argument('--dump-dir', '-du', type=str, default='dumps',
    help='The dir to dump meters/fail-cases/fig/checkpoints')
trainer_args.add_argument('--resume-dir', '-rd', type=str, default=None,
    help='The dir to dump last epoch info that used for resume')
trainer_args.add_argument('--disable-resume', '-dr', action='store_true',
    help='disable resume from saved checkpoint')
trainer_args.add_argument('--dump-fail-cases', '-dfc', action='store_true',
    help='dump fail cases if True')
trainer_args.add_argument('--image-title', '-it', type=str, default=None,
    help='the title of the plot image')
trainer_args.add_argument('--extra', '-ex', type=str, default='',
    help='the extra string for the name of the experiment')
# Misc
trainer_args.add_argument('--exclude-angle-attr', '-ea', action='store_true',
    help='exclude the angle attr when predicting symbolic representation')
trainer_args.add_argument('--key-attr-only', '-ko', action='store_true',
    help='only the key attrs(color/size/type) are considered during observation')
trainer_args.add_argument('--tsne-key', '-tk', type=str, default=None,
    choices=['latent_logits', 'sgm_inter_1', 'sgm_inter_2'],
    help='key name to be visualized by tsne')
trainer_args.add_argument('--tsne-thresh', '-tst', type=float, default=None,
    help='the thresh of ind features to be considered')
trainer_args.add_argument('--tsne-fit-output-file', '-tso', type=str, 
    default='tsne_fit_results.pkl',
    help='the name of saved tsne fit results file')
trainer_args.add_argument('--tsne-positive-k', '-tsp', action='store_true',
    help='record positive k only for tsne if True')

model_args = parser.add_argument_group('Model Related Args')
# Basic
model_args.add_argument('--model', '-m', type=str, default='analogy',
    choices=['simple', 'shared', 'analogy'], help='which model to be used')
model_args.add_argument('--normal-group-mlp', '-ngm', action='store_true',
    help='use normal group mlp instead of shared group mlp if True')
model_args.add_argument('--num-features', '-nf', type=int, default=None,
    help='The number of feature dimensions in the given task')
model_args.add_argument('--one-hot', '-oh', action='store_true',
    help='use one hot representation for attributes inputs if True')
model_args.add_argument('--embedding-hidden-dims', '-ehd', type=int, nargs='+',
    default=[],
    help='the hidden dimensions of the mlp model for embedding (default: [])')
model_args.add_argument('--embedding-dim', '-ed', type=int, default=None,
    help='the dimension of the embedding of features')
model_args.add_argument('--enable-residual-block', '-rb', action='store_true',
    help='use the residual block after embedding if True')
model_args.add_argument('--use-ordinary-mlp', '-om', action='store_true',
    help='use the ordinary mlp instead shared group mlp if True')
model_args.add_argument('--enable-rb-after-experts', '-erb', action='store_true',
    help='use the residual block after experts if True')
# make sure it divs num features or embedding dims (when ed is not None)
model_args.add_argument('--feature-embedding-dim', '-fed', type=int, default=1,
    help='The dimension of embedding space of features')
model_args.add_argument('--num-experts', '-ne', type=int, default=5,
    help='The number of experts used')
# In simple/shared model, this is used as hidden dims of the mlp layer.
model_args.add_argument('--hidden-dims', '-hd', type=int, nargs='+',
    default=[32, 16],
    help='the hidden dimensions of the expert model (default: [32, 16])')
model_args.add_argument('--v2s-softmax', '-vss', action='store_true',
    help='use the softmax after the output of visual mlp if True')
model_args.add_argument('--not-use-softmax', '-ns', action='store_true',
    help='not use the softmax after the output of experts if True')
model_args.add_argument('--reduction-groups', '-rg', type=int, nargs='+',
    default=[2], help='the reduction groups')
model_args.add_argument('--sum-as-reduction', '-sum', type=int, default=0,
    help='how many reductions use sum (only a suffix of reductions)')
model_args.add_argument('--lastmlp-hidden-dims', '-lhd', type=int, nargs='+',
    default=[],
    help='the hidden dimensions of the last mlp model (default: [])')

# Visual
model_args.add_argument('--use-visual-inputs', '-v', action='store_true',
    help='Use visual inputs if True')
model_args.add_argument('--image-size', '-is', type=int, nargs='+',
    default=[160], help='the size of the image')
model_args.add_argument('--use-resnet', '-ur', action='store_true',
    help='use resnet instead of convnets')
model_args.add_argument('--conv-hidden-dims', '-chd', type=int, nargs='+',
    default=[8, 16, 32, 32],
    help='the hidden dimensions of the conv model (default: [8, 16, 32, 32])')
model_args.add_argument('--conv-repeats', '-cr', type=int, nargs='+',
    default=None,
    help='the repeat times of conv layers in each block (default: None)')
model_args.add_argument('--conv-kernels', '-ck', type=int, nargs='+', default=3,
    help='the kernel size of conv layers in each block (default: 3)')
model_args.add_argument('--conv-residual-link', '-crl', action='store_true',
    help='enable the residual links in the conv block')
model_args.add_argument('--use-layer-norm', '-ln', action='store_true',
    help='Use layer norm instead of batch norm for convs')
model_args.add_argument('--visual-mlp-hidden-dims', '-vhd', type=int, nargs='+',
    default=[],
    help='the hidden dimensions of the visual mlp model (default: [])')
model_args.add_argument('--num-visual-experts', '-nve', type=int, default=1,
    help='the number of experts for visual feature extraction')
model_args.add_argument('--factor-groups', '-fg', type=int, default=1,
    help='the groups used in visual model object-feature decomposition')
model_args.add_argument('--split-channel', '-sc', action='store_true',
    help='split the channel dim instead of spatial dim for the output of conv')
model_args.add_argument('--transformed-spatial-dim', '-tsd', type=int,
    default=None, help='the output dim of mlp_transform in visual model')

model_args.add_argument('--mlp-transform-hidden-dims', '-thd',
    type=int, nargs='+', default=[],
    help='the hidden dimensions of the mlp transform model (default: [])')
# Aux Loss
model_args.add_argument('--entropy-beta', '-en', type=float, default=0.0,
    help='To control the weight of entropy loss')
model_args.add_argument('--symbolic-beta', '-sb', type=float, default=0.0,
    help='To control the weight of symbolic loss')
model_args.add_argument('--prediction-beta', '-pb', type=float, default=1.0,
    help='To control the weight of prediction loss')

logger = get_logger(__file__)

args = parser.parse_args()

if args.random_seed is not None:
    random.seed(args.random_seed)
if args.numpy_random_seed is not None:
    np.random.seed(args.numpy_random_seed)
if args.torch_random_seed is not None:
    torch.random.manual_seed(args.torch_random_seed)
    torch.cuda.random.manual_seed_all(args.torch_random_seed)
    # Control Randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# To get reproducible results, set the random seed for torch and numpy
# If there are still inconsistence, set worker_init_fn for the dataloader

args.use_gpu = args.use_gpu and torch.cuda.is_available()
if len(args.split) > 0:
    args.split = '_' + args.split
# The name 'split' is not that proper here,
# used to specify different variation of the dataset.
if len(args.image_size) == 1:
    args.image_size.append(args.image_size[0])
args.image_size = args.image_size[:2]
args.tsne = args.tsne_key is not None

args.is_in_out_structure = len(args.task) == 1 and args.task[0] in [
    'in_out', 'in_distri']
args.is_two_part_structure = len(args.task) == 1 and args.task[0] in [
    'left_right', 'up_down', 'in_out', 'in_distri']

if args.num_context is None:
    args.num_context = 8
if args.num_candidates is None:
    args.num_candidates = 8

args.shared_group_mlp = not args.normal_group_mlp
# Better to use w=2 and cpu=4 for visual inputs
# And w=0 and cpu=2 for symbolic inputs
if args.num_workers is None:
    if args.use_visual_inputs:
        args.num_workers = 2
    else:
        args.num_workers = 0


def get_model(args):
    return Model(
        model_name=args.model,
        nr_features=args.num_features,
        nr_experts=args.num_experts,
        shared_group_mlp=args.shared_group_mlp,
        one_hot=args.one_hot,
        v2s_softmax=args.v2s_softmax,
        not_use_softmax=args.not_use_softmax,
        # visual related
        visual_inputs=args.use_visual_inputs,
        factor_groups=args.factor_groups,
        split_channel=args.split_channel,
        image_size=args.image_size,
        use_layer_norm=args.use_layer_norm,
        use_resnet=args.use_resnet,
        conv_hidden_dims=args.conv_hidden_dims,
        conv_repeats=args.conv_repeats,
        conv_kernels=args.conv_kernels,
        conv_residual_link=args.conv_residual_link,
        nr_visual_experts=args.num_visual_experts,
        visual_mlp_hidden_dims=args.visual_mlp_hidden_dims,
        transformed_spatial_dim=args.transformed_spatial_dim,
        mlp_transform_hidden_dims=args.mlp_transform_hidden_dims,
        exclude_angle_attr=args.exclude_angle_attr,
        symbolic_beta=args.symbolic_beta,
        prediction_beta=args.prediction_beta,
        # embedding
        embedding_dim=args.embedding_dim,
        embedding_hidden_dims=args.embedding_hidden_dims,
        enable_residual_block=args.enable_residual_block,
        use_ordinary_mlp=args.use_ordinary_mlp,
        enable_rb_after_experts=args.enable_rb_after_experts,
        feature_embedding_dim=args.feature_embedding_dim,
        # experts/simple/shared
        hidden_dims=args.hidden_dims,
        # reduction
        reduction_groups=args.reduction_groups,
        sum_as_reduction=args.sum_as_reduction,
        lastmlp_hidden_dims=args.lastmlp_hidden_dims,
        # input format
        nr_context=args.num_context,
        nr_candidates=args.num_candidates,
        # TSNE
        collect_inter_key=args.tsne_key)


def get_dataloader(data_dir, mode, index_file=None, suffix='',
        dataset_size=None):
    if index_file is None:
        inds = None
        fname = mode
        fname += suffix
        if len(args.split) > 0:
            fname += args.split
        if dataset_size is not None:
            fname += '_{}'.format(dataset_size)
        if args.use_visual_inputs:
            fname += '_visual'
        if args.index_file_dir is None:
            dataset = load_data(data_dir, fname)

            if mode == 'train' and args.trunc_train_data is not None:
                # to plot the curve of necessary amount to generalize
                # crop the dataset randomly
                if args.trunc_train_data < len(dataset):
                    random.shuffle(dataset)
                    dataset = dataset[:args.trunc_train_data]

            if mode in ['val', 'test'] and args.trunc_val_data is not None:
                # for tsne plot
                if args.trunc_val_data < len(dataset):
                    # random.shuffle(dataset)
                    dataset = dataset[:args.trunc_val_data]
        else:
            index_file = osp.join(args.index_file_dir, fname + '_inds.pkl')

    if index_file is not None:
        dataset = None
        with open(index_file, 'rb') as f:
            inds = pickle.load(f)
        inds = list(map(lambda x: osp.join(data_dir, x), inds))

    if mode == 'train':
        batch_size = args.batch_size
    elif mode in ['val', 'test']:
        batch_size = args.eval_batch_size
    else:
        assert False

    return JacDataLoader(
        RAVENDataset(dataset,
            task=args.task,
            inds=inds,
            use_visual_inputs=args.use_visual_inputs,
            image_size=args.image_size,
            one_hot=args.one_hot,
            adjust_size=args.adjust_size),
        shuffle=(mode=='train'),
        batch_size=batch_size,
        num_workers=args.num_workers)


def main():
    if len(args.task) == 1:
        task = args.task[0]
        suffix = ''
        dataset_name, num_features = get_dataset_name_and_num_features(task)
        # if args.num_features is None,
        # use the num_features defined by the dataset
        if args.num_features is None:
            args.num_features = num_features

        data_dir = osp.join(args.data_dir, dataset_name)
    else:
        data_dir = args.data_dir
        suffix = '_joint_' + '_'.join(args.task)

    exp_name = get_exp_name(args)
    dump_dir = args.dump_dir
    if args.test_only:
        dump_dir = osp.join(dump_dir, 'tests')
    args.dump_dir = osp.join(dump_dir, exp_name)
    print(args.dump_dir) # for multi-exps usage

    os.makedirs(args.dump_dir, exist_ok=True)
    args.ckpt_dir = osp.join(args.dump_dir, 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.resume_dir is not None:
        os.makedirs(args.resume_dir, exist_ok=True)
    args.summary_file = osp.join(args.dump_dir, 'summary.json')
    args.fail_cases_file = None
    if args.dump_fail_cases:
        args.fail_cases_file = osp.join(args.dump_dir, 'fail_cases.json')
    args.figure_file = osp.join(args.dump_dir, 'curve.png')
    args.meta_file = osp.join(args.dump_dir, 'meta.json')
    args.log_file = osp.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)

    # logger.info('\n' + kvformat(args.__dict__))
    logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
    with open(args.meta_file, 'w') as f:
        f.write(dump_metainfo(args=args.__dict__))

    train_dataset_size = None
    val_dataset_size = None
    if args.dataset_size is not None:
        train_dataset_size = args.dataset_size[0]
        val_dataset_size = args.dataset_size[1]
    if not args.test_only:
        train_data = get_dataloader(data_dir, 'train',
            index_file=args.train_index_file, suffix=suffix,
            dataset_size=train_dataset_size)
    val_dataset_name = 'test' if args.use_test_as_val else 'val'
    val_data = get_dataloader(data_dir, val_dataset_name,
        index_file=args.val_index_file, suffix=suffix,
        dataset_size=val_dataset_size)

    model = get_model(args)

    v2s_lr = args.v2s_lr
    if args.use_visual_inputs:
        if v2s_lr is None:
            v2s_lr = args.lr
        param_groups = [
            dict(params=model.v2s.parameters(), lr=v2s_lr),
            dict(params=model.symbolic_model.parameters())]
    else:
        param_groups = model.parameters()
    # optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)
    if args.weight_decay == 0:
        optimizer = optim.Adam(param_groups, lr=args.lr)
    else:
        optimizer = optim.AdamW(param_groups, lr=args.lr,
            weight_decay=args.weight_decay)

    observe_env = None
    if args.observe_interval is not None or args.test_obs:
        # Observer: match the symbolic representation use learned ones
        observe_model = Observer(model.nr_ind_features, num_features,
            feature_embedding_dim=args.feature_embedding_dim,
            exclude_angle_attr=args.exclude_angle_attr,
            key_attr_only=args.key_attr_only,
            thresh=args.obs_thresh,
            exclude_outside_color=args.is_in_out_structure)
        if args.use_gpu:
            observe_model.cuda()
        obs_lr = args.obs_lr
        if obs_lr is None:
            obs_lr = args.lr
        observe_optimizer = optim.Adam(
            observe_model.parameters(), lr=obs_lr)
        observe_env = TrainerEnv(observe_model, observe_optimizer)

    if args.use_gpu:
        # model.cuda()
        model = JacDataParallel(model).cuda()

    trainer = Trainer(model,
        optimizer,
        epochs=args.epochs,
        lr=args.lr,
        use_visual_inputs=args.use_visual_inputs,
        v2s_lr=v2s_lr,
        lr_anneal_start=args.lr_anneal_start,
        lr_anneal_interval=args.lr_anneal_interval,
        lr_anneal_ratio=args.lr_anneal_ratio,
        obs_val_only=args.obs_val_only,
        obs_epochs=args.obs_epochs,
        observe_env=observe_env,
        observe_interval=args.observe_interval,
        use_gpu=args.use_gpu,
        monitor_grads=args.monitor_grads,
        save_interval=args.save_interval,
        ckpt_dir=args.ckpt_dir,
        resume_dir=args.resume_dir,
        disable_resume=args.disable_resume,
        summary_file=args.summary_file,
        fail_cases_file=args.fail_cases_file)
    if args.load is not None:
        if trainer.load_weights(args.load):
            logger.critical(('Loaded weights from pretrained model: '
                '"{}".').format(args.load))

    if args.test_only:
        assert args.load is not None
        extra_dict = {}
        val_meters = trainer.val_epoch(val_data,
            collect_inter_key=args.tsne_key, extra_dict=extra_dict)
        trainer._dump_meters(val_meters, 'val')
        if args.test_obs:
            _, output_dict = trainer.observe(val_data, dataset_name='val')
            inds = output_dict['inds'].cpu().numpy()
            errs = output_dict['errs'].detach().cpu().numpy()
            weight = output_dict['weight'].detach().cpu().numpy()
            if args.tsne:
                k = weight[np.arange(len(inds)), inds]
        elif args.tsne:
            assert False, 'unable to do tsne plot without obs results'

        # TSNE plot for intermediate representation
        if args.tsne:
            inter = extra_dict['inter']
            gts = extra_dict['gts']
            nr_examples = len(gts)
            nr_features = len(inds)
            nr_possible_rels = 4

            # print(inds)
            # print(errs)
            inds = np.array(inds)
            inds_add = 0
            if args.is_two_part_structure:
                inds_add = (inds >= 3).astype('int32') * 3
            inds = inds % 3 + inds_add

            seed = 0
            if args.random_seed is not None:
                seed = args.random_seed
            tsne = TSNE(n_components=2, random_state=seed)

            data = []
            colors = []
            origin = []
            for i in range(nr_features):
                if args.tsne_thresh is None or errs[i] < args.tsne_thresh:
                    if not args.tsne_positive_k or k[i] > 0.:
                        for j in range(nr_examples):
                            c = gts[j, inds[i]]
                            # print('id={}, align={}, rel={}, err={}, k={}'.format(
                            #     i, inds[i], c, errs[i], k[i]))
                            data.append(inter[j, i])
                            colors.append(c)
                            origin.append(j * nr_features + i)
            data = np.array(data)
            logger.info('start TSNE fitting for {} data'.format(len(data)))
            t1 = time.time()
            results = tsne.fit_transform(data)
            logger.info('time usage for TSNE {:.6f}'.format(time.time() - t1))
            with open(args.tsne_fit_output_file, 'wb') as f:
                dump_dict = dict(results=results, colors=colors, origin=origin)
                pickle.dump(dump_dict, f)
                logger.info('TSNE fitting results saved to {}'.format(
                    args.tsne_fit_output_file))
    else:
        trainer.train_eval(train_data, val_data)
        # NOTE: temporarily disable the training curve plot
        # all_meters = trainer.train_eval(train_data, val_data)
        # image_title = get_image_title(args)
        # plot_curve(all_meters, image_title, args.figure_file)


if __name__ == '__main__':
    main()
