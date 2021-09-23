from torch import nn

import abc


class Dynamics(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    # @abc.abstractclassmethod 
    # def eval(self, x, u):
    #     return NotImplementedError()

    # @abc.abstractclassmethod 
    # def eval_sens(self, x, u):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_hess(self, x, u, lmd):
    #     return NotImplementedError()

    # @abc.abstractclassmethod
    # def eval_param_sens(self, x, u):
    #     return NotImplementedError()
