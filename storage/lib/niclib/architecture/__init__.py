from abc import ABC, abstractmethod

import torch

from niclib.network.layers import DropoutPrediction

class NIC_Architecture(ABC, torch.nn.Module):
    def activate_dropout_testing(self, p_out, dotype):
        def activate_dropout(m):
            if isinstance(m, DropoutPrediction):
                m.activate(p_out, dotype)
        self.apply(activate_dropout)

    def deactivate_dropout_testing(self):
        def deactivate_dropout(m):
            if isinstance(m, DropoutPrediction):
                m.deactivate()
        self.apply(deactivate_dropout)


def load_architecture():
    pass