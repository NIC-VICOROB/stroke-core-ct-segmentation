import copy

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Poisson

class DropoutPrediction(nn.Module):
    def __init__(self, inactive=False, ndims=3):
        """
        :param inactive: if True, layer cannot be activated, this avoids changing network design if dropout is not required
        :param ndims:
        """
        super().__init__()

        self.channel_dropout = nn.Dropout2d if ndims is 2 else nn.Dropout3d
        self.alpha_dropout = nn.AlphaDropout

        self.forever_inactive = inactive
        self._running = False
        self.d = self.channel_dropout(p=0.5)

    def forward(self, x_in):
        if not self._running or self.forever_inactive: self.d.eval()
        else: self.d.train()

        return self.d(x_in)

    def activate(self, p_out=None, dotype='channel'):
        self._running = True
        if dotype is 'channel':
            self.d = self.channel_dropout(p=p_out)
        elif dotype is 'alpha':
            self.d = self.alpha_dropout(p=p_out)
        else:
            raise NotImplementedError

    def deactivate(self):
        self._running = False

#import visdom
#viz = visdom.Visdom()
class LearnableAverage(torch.nn.Module):
    def __init__(self, num_elements):
        super().__init__()
        self.input_weights = torch.nn.Parameter(torch.tensor([1.0 / num_elements] * num_elements))
        self.X = torch.tensor([0.0])

    def forward(self, input_list):
        input_weighted = torch.stack(
            [input_original * self.input_weights[i] for i, input_original in enumerate(input_list)], dim=0)
        return torch.sum(input_weighted, dim=0) / torch.sum(self.input_weights)




class CTNoiser(torch.nn.Module):
    """
    Module for additive CT image noise, a mixture of Gaussian and Poisson
    """
    def __init__(self, mean, std, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.noise_distribution = Normal(loc=mean, scale=std)

    def forward(self, x_in):
        return x_in + self.noise_distribution.sample(x_in.size()).to(self.device)


class CTNoiserParametric(torch.nn.Module):
    """
    Module for additive CT image noise, a mixture of Gaussian and Poisson
    """
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device

        self.normal_scale = torch.nn.Parameter(torch.tensor(np.abs(np.random.randn()*0.02)))
        self.normal_mean = torch.nn.Parameter(torch.tensor(np.abs(np.random.rand()*0.02)))
        self.normal_std = torch.nn.Parameter(torch.tensor(np.abs(np.random.rand()*0.05 + 0.05)))

    def forward(self, x_in):
        normal_noise = ((Normal(0.0, 1.0).sample(x_in.size()).to(self.device) * self.normal_std) + self.normal_mean)
        return x_in + self.normal_scale * normal_noise

