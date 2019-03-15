import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from niclib.architecture import NIC_Architecture
from niclib.network.layers import DropoutPrediction


class Unet3D(NIC_Architecture):
    def __init__(self, in_ch=1, out_ch=1, ndims=3, nfilts = 16, softmax_out=True):
        super().__init__()

        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d
        BatchNorm = nn.BatchNorm2d if ndims is 2 else nn.BatchNorm3d
        MaxPool = nn.MaxPool2d if ndims is 2 else nn.MaxPool3d

        self.conv1 = torch.nn.Sequential(
            Conv(in_ch, nfilts, 3, padding=1), BatchNorm(nfilts, momentum=0.01), nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            Conv(nfilts, 2 * nfilts, 3, padding=1), BatchNorm(2 * nfilts, momentum=0.01), nn.ReLU())
        self.down1 = MaxPool(2)

        self.conv3 = torch.nn.Sequential(
            Conv(2 * nfilts, 2 * nfilts, 3, padding=1), BatchNorm(2 * nfilts, momentum=0.01), nn.ReLU())
        self.conv4 = torch.nn.Sequential(
            Conv(2 * nfilts, 4 * nfilts, 3, padding=1), BatchNorm(4 * nfilts, momentum=0.01), nn.ReLU())
        self.down2 = MaxPool(2)

        self.conv5 = torch.nn.Sequential(
            DropoutPrediction(), Conv(4 * nfilts, 4 * nfilts, 3, padding=1), BatchNorm(4 * nfilts, momentum=0.01), nn.ReLU())
        self.conv6 = torch.nn.Sequential(
            DropoutPrediction(), Conv(4 * nfilts, 8 * nfilts, 3, padding=1), BatchNorm(8 * nfilts, momentum=0.01), nn.ReLU())
        self.down3 = MaxPool(2)

        self.conv7 = torch.nn.Sequential(
            DropoutPrediction(), Conv(8 * nfilts, 8 * nfilts, 3, padding=1), BatchNorm(8 * nfilts, momentum=0.01), nn.ReLU())
        self.conv8 = torch.nn.Sequential(
            DropoutPrediction(), Conv(8 * nfilts, 16 * nfilts, 3, padding=1), BatchNorm(16 * nfilts, momentum=0.01), nn.ReLU())
        self.upconv1 = ConvTranspose(16 * nfilts, 16 * nfilts, 3, padding=1, output_padding=1, stride=2)

        self.conv9 = torch.nn.Sequential(
            DropoutPrediction(), Conv(16 * nfilts + 8 * nfilts, 8 * nfilts, 3, padding=1), BatchNorm(8 * nfilts, momentum=0.01), nn.ReLU())
        self.conv10 = torch.nn.Sequential(
            DropoutPrediction(), Conv(8 * nfilts, 8 * nfilts, 3, padding=1), BatchNorm(8 * nfilts, momentum=0.01), nn.ReLU())
        self.upconv2 = ConvTranspose(8 * nfilts, 8 * nfilts, 3, padding=1, output_padding=1, stride=2)

        self.conv11 = torch.nn.Sequential(
            Conv(8 * nfilts + 4 * nfilts, 4 * nfilts, 3, padding=1), BatchNorm(4 * nfilts, momentum=0.01), nn.ReLU())
        self.conv12 = torch.nn.Sequential(
            Conv(4 * nfilts, 4 * nfilts, 3, padding=1), BatchNorm(4 * nfilts, momentum=0.01), nn.ReLU())
        self.upconv3 = ConvTranspose(4 * nfilts, 4 * nfilts, 3, padding=1, output_padding=1, stride=2)

        self.conv13 = torch.nn.Sequential(Conv(4 * nfilts + 2 * nfilts, 2 * nfilts, 3, padding=1), BatchNorm(2 * nfilts, momentum=0.01), nn.ReLU())
        self.conv14 = torch.nn.Sequential(Conv(2 * nfilts, 2 * nfilts, 3, padding=1), BatchNorm(2 * nfilts, momentum=0.01), nn.ReLU())

        self.out_conv = Conv(2 * nfilts, out_ch, 3, padding=1)
        self.softmax_out = nn.Softmax(dim=1) if softmax_out is True else None

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("Unet3D (Cicek) network with {} parameters".format(nparams))

    def forward(self, x_in):
        l1_start = self.conv1(x_in)
        l1_end = self.conv2(l1_start)

        l2_start = self.down1(l1_end)
        l2_mid = self.conv3(l2_start)
        l2_end = self.conv4(l2_mid)

        l3_start = self.down2(l2_end)
        l3_mid = self.conv5(l3_start)
        l3_end = self.conv6(l3_mid)

        l4_start = self.down3(l3_end)
        l4_mid = self.conv7(l4_start)
        l4_end = self.conv8(l4_mid)

        r3_start = torch.cat((l3_end, self.upconv1(l4_end)), dim=1)
        r3_mid = self.conv9(r3_start)
        r3_end = self.conv10(r3_mid)

        r2_start = torch.cat((l2_end, self.upconv2(r3_end)), dim=1)
        r2_mid = self.conv11(r2_start)
        r2_end = self.conv12(r2_mid)

        r1_start = torch.cat((l1_end, self.upconv3(r2_end)), dim=1)
        r1_mid = self.conv13(r1_start)
        r1_end = self.conv14(r1_mid)

        x_out = self.out_conv(r1_end)
        if self.softmax_out is not None:
            x_out = self.softmax_out(x_out)

        return x_out