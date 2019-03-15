from abc import ABC, abstractmethod

class NICOptimizer(ABC):
    @abstractmethod
    def set_parameters(self, params):
        pass


class TorchOptimizer(NICOptimizer):
    """
    Wrapper for configuring default torch optimizers before assigning model parameters to optimize
    """
    def __init__(self, optim_class, opts=None):
        self.optim_class = optim_class
        self.optim_options = opts

    def set_parameters(self, params):
        return self.optim_class(params, **self.optim_options) if self.optim_options else self.optim_class(params)