from torch import nn

import abc


class TerminalCost(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    # @abc.abstractclassmethod
    # def eval(self, x):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_sens(self, x):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_hess(self, x):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_param_sens(self, x):
    #     return NotImplementedError()


class StageCost(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    # @abc.abstractclassmethod
    # def eval(self, x, u, stage):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_sens(self, x, u, stage):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_hess(self, x, u, stage):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_param_sens(self, x, u, stage):
    #     return NotImplementedError()