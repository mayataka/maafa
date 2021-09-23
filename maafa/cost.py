from torch import nn

import abc


class TerminalCost(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()


class StageCost(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
