#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : thutils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 11/22/2019
#
# Distributed under terms of the MIT license.

# copied from anonymous place

import torch
import torch.nn.functional as F

from jactorch.utils.meta import as_tensor, as_float

__all__ = ['rms', 'monitor_saturation', 'monitor_paramrms', 'monitor_gradrms']


def rms(p):
    return as_float((as_tensor(p) ** 2).mean() ** 0.5)


def monitor_saturation(model):
    monitors = {}
    for name, p in model.named_parameters():
        p = F.sigmoid(p)
        sat = 1 - (p - (p > 0.5).float()).abs()
        monitors['sat/' + name] = sat
    return monitors


def monitor_paramrms(model):
    monitors = {}
    for name, p in model.named_parameters():
        monitors['paramrms/' + name] = rms(p)
    return monitors


def monitor_gradrms(model):
    monitors = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            monitors['gradrms/' + name] = rms(p.grad) / max(rms(p), 1e-8)
    return monitors
