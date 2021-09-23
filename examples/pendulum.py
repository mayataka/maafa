import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

import maafa
from pendulum.dynamics import PendulumDynamics
from pendulum.cost import PendulumTerminalCost, PendulumStageCost

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')


if __name__ == '__main__':
    T = 1.0
    N = 20
    dt = T / N
    gamma = 0.99
    nbatch = 10
    dynamics = PendulumDynamics(dt)
    terminal_cost = PendulumTerminalCost()
    stage_cost = PendulumStageCost(dt, gamma)
    ocp = maafa.ocp.OCP(dynamics, stage_cost, terminal_cost, N)
    x = torch.rand(N+1, nbatch, dynamics.dimx)
    u = torch.rand(N, nbatch, dynamics.dimu)
    lmd = torch.rand(N+1, nbatch, dynamics.dimx)
    x0 = torch.rand(dynamics.dimx)
    x0res = x[0] - x0
    kkt = ocp.eval_kkt(x0, x, u, lmd)

    riccati = maafa.riccati_recursion.RiccatiRecursion(dynamics, N)
    dx, du, dlmd = riccati.riccati_recursion(kkt)
