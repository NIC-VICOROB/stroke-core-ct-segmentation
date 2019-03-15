import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from niclib.architecture import NIC_Architecture
from niclib.network.layers import DropoutPrediction

INIT_PRELU = 0.0
BN_MOMENTUM = 0.01


class SUNETx4(NIC_Architecture):
    def __init__(self, in_ch=5, out_ch=2, nfilts=32, ndims=3, softmax_out=True, dropout_rate=0.2):
        super(SUNETx4, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d

        self.inconv = Conv(in_ch, 1 * nfilts, 3, padding=1)

        self.dual1 = DualRes(1 * nfilts, ndims)
        self.dual2 = DualRes(2 * nfilts, ndims, dropout_rate=dropout_rate, dropout_prediction=True)
        self.dual3 = DualRes(4 * nfilts, ndims, dropout_rate=dropout_rate, dropout_prediction=True)
        self.dual4 = DualRes(8 * nfilts, ndims, dropout_rate=dropout_rate, dropout_prediction=True)

        self.down1 = DownStep(1 * nfilts, ndims)
        self.down2 = DownStep(2 * nfilts, ndims)
        self.down3 = DownStep(4 * nfilts, ndims)

        self.mono3 = MonoRes(4 * nfilts, ndims, dropout_prediction=True)
        self.mono2 = MonoRes(2 * nfilts, ndims, dropout_prediction=True)
        self.mono1 = MonoRes(1 * nfilts, ndims)

        self.up4 = ConvTranspose(in_channels=8 * nfilts, out_channels=4 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.up3 = ConvTranspose(in_channels=4 * nfilts, out_channels=2 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.up2 = ConvTranspose(in_channels=2 * nfilts, out_channels=1 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)

        self.outconv = Conv(nfilts, out_ch, 3, padding=1)

        self.do_softmax = softmax_out
        self.softmax = nn.Softmax(dim=1) # Channels dimension

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("SUNETx4_{}D_f{} network with {} parameters".format(ndims, nfilts, nparams))

    def forward(self, x_in):
        l1_start = self.inconv(x_in)

        l1_end = self.dual1(l1_start)
        l2_start = self.down1(l1_end)

        l2_end = self.dual2(l2_start)
        l3_start = self.down2(l2_end)

        l3_end = self.dual3(l3_start)
        l4_start = self.down3(l3_end)

        l4_latent = self.dual4(l4_start)
        r4_up = self.up4(l4_latent)

        r3_start = l3_end + r4_up
        r3_end = self.mono3(r3_start)
        r3_up = self.up3(r3_end)

        r2_start = l2_end + r3_up
        r2_end = self.mono2(r2_start)
        r2_up = self.up2(r2_end)

        r1_start = l1_end + r2_up
        r1_end = self.mono1(r1_start)

        pred = self.outconv(r1_end)
        if self.do_softmax:
            pred = self.softmax(pred)

        return pred


class SUNETx5(NIC_Architecture):
    def __init__(self, in_ch=5, out_ch=2, nfilts=32, ndims=3, dropout_rate=0.2):
        super(SUNETx5, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d

        self.inconv = Conv(in_ch, 1 * nfilts, 3, padding=1)

        self.dual1 = DualRes(1 * nfilts, ndims)
        self.dual2 = DualRes(2 * nfilts, ndims)
        self.dual3 = DualRes(4 * nfilts, ndims, dropout_rate=dropout_rate, dropout_prediction=True)
        self.dual4 = DualRes(8 * nfilts, ndims, dropout_rate=dropout_rate, dropout_prediction=True)
        self.dual5 = DualRes(16 * nfilts, ndims, dropout_rate=dropout_rate, dropout_prediction=True)

        self.down1 = DownStep(1 * nfilts, ndims)
        self.down2 = DownStep(2 * nfilts, ndims)
        self.down3 = DownStep(4 * nfilts, ndims)
        self.down4 = DownStep(8 * nfilts, ndims)

        self.mono4 = MonoRes(8 * nfilts, ndims, dropout_prediction=True)
        self.mono3 = MonoRes(4 * nfilts, ndims, dropout_prediction=True)
        self.mono2 = MonoRes(2 * nfilts, ndims)
        self.mono1 = MonoRes(1 * nfilts, ndims)

        self.up5 = ConvTranspose(in_channels=16 * nfilts, out_channels=8 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.up4 = ConvTranspose(in_channels=8 * nfilts, out_channels=4 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.up3 = ConvTranspose(in_channels=4 * nfilts, out_channels=2 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.up2 = ConvTranspose(in_channels=2 * nfilts, out_channels=1 * nfilts, kernel_size=3, padding=1, output_padding=1, stride=2)

        self.outconv = Conv(nfilts, out_ch, 3, padding=1)
        self.softmax = nn.Softmax(dim=1) # Channels dimension

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("SUNETx5_{}D_f{} network with {} parameters".format(ndims, nfilts, nparams))

    def forward(self, x_in):
        l1_start = self.inconv(x_in)

        l1_end = self.dual1(l1_start)
        l2_start = self.down1(l1_end)

        l2_end = self.dual2(l2_start)
        l3_start = self.down2(l2_end)

        l3_end = self.dual3(l3_start)
        l4_start = self.down3(l3_end)

        l4_end = self.dual4(l4_start)
        l5_start = self.down4(l4_end)

        l5_latent = self.dual5(l5_start)
        r5_up = self.up5(l5_latent)

        r4_start = l4_end + r5_up
        r4_end = self.mono4(r4_start)
        r4_up = self.up4(r4_end)

        r3_start = l3_end + r4_up
        r3_end = self.mono3(r3_start)
        r3_up = self.up3(r3_end)

        r2_start = l2_end + r3_up
        r2_end = self.mono2(r2_start)
        r2_up = self.up2(r2_end)

        r1_start = l1_end + r2_up
        r1_end = self.mono1(r1_start)

        pred = self.outconv(r1_end)
        pred = self.softmax(pred)

        return pred

class DualRes(nn.Module):
    def __init__(self, num_ch, ndims=3, dropout_rate=0.0, dropout_prediction=False):
        super(DualRes, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d
        Dropout = nn.Dropout2d if ndims is 2 else nn.Dropout3d

        self.conv_path = nn.Sequential(
            DropoutPrediction(inactive=not dropout_prediction),
            BatchNorm(num_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(num_ch, init=INIT_PRELU),
            Conv(num_ch, num_ch, 3, padding=1),
            Dropout(p=dropout_rate),
            BatchNorm(num_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(num_ch, init=INIT_PRELU),
            Conv(num_ch, num_ch, 3, padding=1))

    def forward(self, x_in):
        return self.conv_path(x_in) + x_in


class MonoRes(nn.Module):
    def __init__(self, num_ch, ndims=3, dropout_prediction=False):
        super(MonoRes, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d

        self.conv_path = nn.Sequential(
            DropoutPrediction(inactive=not dropout_prediction),
            BatchNorm(num_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(num_ch, init=INIT_PRELU),
            Conv(num_ch, num_ch, 3, padding=1))

    def forward(self, x_in):
        x_out = self.conv_path(x_in) + x_in
        return x_out


class DownStep(nn.Module):
    def __init__(self, in_ch, ndims=3):
        super(DownStep, self).__init__()
        MaxPool = nn.MaxPool2d if ndims is 2 else nn.MaxPool3d
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d

        self.pool_path = MaxPool(2)
        self.conv_path = nn.Sequential(
            BatchNorm(in_ch, momentum=BN_MOMENTUM, eps=0.001),
            nn.PReLU(in_ch, init=INIT_PRELU),
            Conv(in_ch, in_ch, 3, padding=1, stride=2))

    def forward(self, x_in):
        x_out = torch.cat((self.conv_path(x_in), self.pool_path(x_in)), dim=1) # Channel dimension
        return x_out









