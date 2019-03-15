import warnings

import torch
import torch.nn as nn
import numpy as np
from niclib.network.layers import LearnableAverage



# import visdom
# viz = visdom.Visdom()
#
# def draw_wavg_weights(m):
#     if isinstance(m, Ensemble_WAVG):
#         weights = m.average_conv.weight.data.clone().detach().cpu()
#         bias = m.average_conv.bias.data.clone().detach().cpu()
#
#         weights0 = torch.reshape(weights[0,...], (1, -1))
#         weights1 = torch.reshape(weights[1,...], (1, -1))
#         bias0 = torch.reshape(bias[0], (1, -1))
#         bias1 = torch.reshape(bias[1], (1, -1))
#
#         viz.line(Y=weights0, X = m.x * torch.ones_like(weights0), update='append', win='weights0', opts=dict(title='weights0'))
#         viz.line(Y=weights1, X = m.x * torch.ones_like(weights1), update='append', win='weights1', opts=dict(title='weights1'))
#         viz.line(Y=bias0, X = m.x * torch.ones_like(bias0), update='append', win='bias0', opts=dict(title='bias0'))
#         viz.line(Y=bias1, X = m.x * torch.ones_like(bias1), update='append', win='bias1', opts=dict(title='bias1'))
#         m.x += 1

def init_wavg_weights(m):
    if isinstance(m, Ensemble_WAVG):
        nn.init.constant_(m.average_conv.weight, 1.0)
        nn.init.constant_(m.average_conv.bias, 0.0)

class Ensemble_WAVG(torch.nn.Module):
    def __init__(self, models, num_classes, ndims):
        assert isinstance(models, list)
        super().__init__()
        Conv = nn.Conv2d if ndims is 2 else nn.Conv3d
        self.x = 0

        # Freeze weights
        for model in models:
            if model.__class__.__name__ in {'SUNETx4', 'SUNETx5', 'uResNet'}:
                model.do_softmax = False
            elif model.__class__.__name__ in {'Unet3D'}:
                model.softmax_out = None
            else:
                warnings.warn("Couldn't disable softmax output of model, please disble manually")

            for params in model.parameters():
                params.requires_grad = False

        self.models = torch.nn.ModuleList(models)
        self.average_conv = Conv(len(models) * num_classes, num_classes, 1)
        self.softmax_out = torch.nn.Softmax(dim=1)

        self.apply(init_wavg_weights)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print("Ensemble_WAVG network with {} trainable parameters".format(nparams))

    def forward(self, x_in):
        x_avg = self.average_conv(torch.cat([m(x_in) for m in self.models], dim=1))
        return self.softmax_out(x_avg)



