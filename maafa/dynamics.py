from torch import nn

import abc


class Dynamics(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
