import torch
from niclib.architecture import NIC_Architecture

class Ensemble_AVG(NIC_Architecture):
    def __init__(self, models, device=torch.device('cuda')):
        assert isinstance(models, list)
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        print("Making Ensemble_AVG model with {} models".format(len(models)))

    def forward(self, x_in):
        return torch.mean(torch.stack([m(x_in) for m in self.models], dim=0), dim=0)


