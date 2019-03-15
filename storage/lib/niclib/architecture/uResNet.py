import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from niclib.architecture import NIC_Architecture
from niclib.network.layers import DropoutPrediction

class uResNet(NIC_Architecture):
    def __init__(self, in_ch, out_ch, ndims=2, nfilts=32, do_softmax=True):
        super().__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d
        MaxPool = nn.MaxPool2d if ndims is 2 else nn.MaxPool3d

        self.enc_res1 = ResEle(in_ch, nfilts, ndims=ndims)
        self.enc_res2 = ResEle(1 * nfilts, 2 * nfilts, ndims=ndims)
        self.enc_res3 = ResEle(2 * nfilts, 4 * nfilts, ndims=ndims, dropout_prediction=True)
        self.enc_res4 = ResEle(4 * nfilts, 8 * nfilts, ndims=ndims, dropout_prediction=True)

        self.pool1 = MaxPool(2)
        self.pool2 = MaxPool(2)
        self.pool3 = MaxPool(2)

        self.dec_res1 = ResEle(4 * nfilts, 4 * nfilts, ndims=ndims, dropout_prediction=True)
        self.dec_res2 = ResEle(4 * nfilts, 2 * nfilts, ndims=ndims, dropout_prediction=True)
        self.dec_res3 = ResEle(2 * nfilts, 2 * nfilts, ndims=ndims)
        self.dec_res4 = ResEle(1 * nfilts, 1 * nfilts, ndims=ndims)

        self.deconv1 = ConvTranspose(8 * nfilts, 4 * nfilts, 3, padding=1, output_padding=1, stride=2)
        self.deconv2 = ConvTranspose(2 * nfilts, 2 * nfilts, 3, padding=1, output_padding=1, stride=2)
        self.deconv3 = ConvTranspose(2 * nfilts, 1 * nfilts, 3, padding=1, output_padding=1, stride=2)

        self.out_conv = Conv(nfilts, out_ch, 1)
        self.do_softmax = do_softmax
        self.out_softmax = nn.Softmax(dim=1)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("uResNet_{}D_{}f network with {} parameters".format(ndims, nfilts, nparams))

    def forward(self, x_in):
        l1_end = self.enc_res1(x_in)

        l2_start = self.pool1(l1_end)
        l2_end = self.enc_res2(l2_start)

        l3_start = self.pool2(l2_end)
        l3_end = self.enc_res3(l3_start)

        l4_start = self.pool3(l3_end)
        l4_end = self.enc_res4(l4_start)

        r4_start = self.deconv1(l4_end)
        r4_end = self.dec_res1(r4_start)

        r3_start = self.dec_res2(r4_end + l3_end)
        r3_end = self.deconv2(r3_start)

        r2_start = self.dec_res3(r3_end + l2_end)
        r2_end = self.deconv3(r2_start)

        r1_start = self.dec_res4(r2_end + l1_end)
        pred = self.out_conv(r1_start)

        if self.do_softmax:
            pred = self.out_softmax(pred)

        return pred


class ResEle(torch.nn.Module):
    def __init__(self, nch_in, nch_out, ndims=2, dropout_prediction=False):
        super().__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d

        self.selection_path = Conv(nch_in, nch_out, 1)
        self.conv_path = Conv(nch_in, nch_out, 3, padding=1)
        self.output_path = nn.Sequential(
            DropoutPrediction(inactive=not dropout_prediction),
            BatchNorm(nch_out, momentum=0.01, eps=0.001),
            nn.ReLU())

    def forward(self, x_in):
        return self.output_path(self.conv_path(x_in) + self.selection_path(x_in))