import torch
from niclib.architecture import NIC_Architecture


class Ensemble_TEST(NIC_Architecture):
    def __init__(self, filepaths, device=torch.device('cuda')):
        super().__init__()
        print("Ensemble_TEST: Loading models")
        for fp in filepaths:
            print(fp)

        self.models = torch.nn.ModuleList([torch.load(fp, device) for fp in filepaths])

    def forward(self, x_in):
        return torch.mean(torch.stack([m(x_in) for m in self.models], dim=0), dim=0)


