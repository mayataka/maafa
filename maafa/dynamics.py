from torch import nn

import abc


class Dynamics(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    @abc.abstractclassmethod
    def eval(self, x, u, x_res):
        return NotImplementedError()

    @abc.abstractclassmethod
    def eval_sens(self, x, u, A, B):
        return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_hess(self, x, u, lmd, Qxx, Qxu, Quu):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_param_sens(self, x, u, dp):
    #     return NotImplementedError()
