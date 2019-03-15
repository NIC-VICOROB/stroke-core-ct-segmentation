import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


import warnings

import numpy as np

from niclib.architecture.SUNet import SUNETx4, SUNETx5
from niclib.network.layers import CTNoiser

import visdom
viz = visdom.Visdom()

class AutoDenoiser(torch.nn.Module):
    def __init__(self, noiser_module, denoiser_module, segmenter_module):
        super().__init__()
        warnings.warn("AutoDenoiser requires first modality (C=0) to be CT in an image of dims (B,C,X,Y,Z)")
        self.noiser = noiser_module
        self.denoiser = denoiser_module
        self.segmenter = segmenter_module

        self.X = torch.tensor([0.0])

        self.vis_interval = 20
        self.vis_counter = 0

    def forward(self, x_in):
        x_ct = x_in[:, :1, ...]
        x_rest = x_in[:, 1:, ...]

        return x_in






