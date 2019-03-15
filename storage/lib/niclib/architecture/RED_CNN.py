import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RED_CNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ndims=2, nfilts=96):
        super(RED_CNN, self).__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if ndims is 2 else nn.ConvTranspose3d

        self.in_conv = nn.Sequential(
            Conv(in_ch, nfilts, 5),
            nn.ReLU())

        self.encoder_blocks = torch.nn.ModuleList([nn.Sequential(Conv(nfilts, nfilts, 5), nn.ReLU()) for _ in range(4)])

        self.deconvs = torch.nn.ModuleList([ConvTranspose(nfilts, nfilts, 5) for _ in range(4)])
        self.out_conv = ConvTranspose(nfilts, out_ch, 5)
        self.relus = torch.nn.ModuleList([nn.ReLU() for _ in range(5)])

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("RED-CNN network with {} parameters".format(nparams))

    def forward(self, x_in):
        e1 = self.in_conv(x_in)
        e2 = self.encoder_blocks[0](e1)
        e3 = self.encoder_blocks[1](e2)
        e4 = self.encoder_blocks[2](e3)
        e5 = self.encoder_blocks[3](e4)

        dec1_in = self.deconvs[0](e5)
        dec1_out = self.relus[0](dec1_in + e4)

        dec2_in = self.deconvs[1](dec1_out)
        dec2_out = self.relus[1](dec2_in)

        dec3_in = self.deconvs[2](dec2_out)
        dec3_out = self.relus[2](dec3_in + e2)

        dec4_in = self.deconvs[3](dec3_out)
        dec4_out = self.relus[3](dec4_in)

        dec5_in = self.out_conv(dec4_out)
        dec5_out = self.relus[4](dec5_in + x_in)

        return  dec5_out


