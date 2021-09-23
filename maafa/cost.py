from torch import nn

import abc


class TerminalCost(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractclassmethod
    def eval(self, x):
        return NotImplementedError()

    @abc.abstractclassmethod
    def eval_sens(self, x, lx):
        return NotImplementedError()

    @abc.abstractclassmethod
    def eval_hess(self, x, Qxx):
        return NotImplementedError()

    @abc.abstractclassmethod
    def eval_param_sens(self, x, dp):
        return NotImplementedError()


class StageCost(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    @abc.abstractclassmethod
    def eval(self, x, u):
        return NotImplementedError()

    @abc.abstractclassmethod
    def eval_sens(self, x, u, lx, lu):
        return NotImplementedError()

    @abc.abstractclassmethod
    def eval_hess(self, x, u, Qxx, Qxu, Quu):
        return NotImplementedError()

    @abc.abstractclassmethod
    def eval_param_sens(self, x, u, dp):
        return NotImplementedError()