import torch
from torch.autograd import Function, Variable
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np
import math

from .riccati_recursion import RiccatiRecursion
from .kkt import KKT
from . import utils


class MPC(nn.Module):
    def __init__(self, dynamics, stage_cost, terminal_cost, N, GaussNewton=True, nbatch=1):
        super().__init__()
        self.dynamics = dynamics
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.nx = dynamics.nx
        self.nu = dynamics.nu
        self.N = N
        self.GaussNewton = GaussNewton
        self.nbatch = nbatch
        self.kkt = KKT(dynamics, N, nbatch)
        self.riccati_recursion = RiccatiRecursion(dynamics, N, nbatch)
        nx = dynamics.nx()
        nu = dynamics.nu()
        self.x = torch.zeros(N+1, nbatch, nx) 
        self.u = torch.zeros(N, nbatch, nu) 
        self.lmd = torch.zeros(N+1, nbatch, nx) 
