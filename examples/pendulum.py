import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

import maafa
from pendulum.dynamics import PendulumDynamics
from pendulum.cost import PendulumTerminalCost, PendulumStageCost


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')


if __name__ == '__main__':
    T = 2.0
    N = 20
    dt = T / N
    gamma = 1.0
    nbatch = 10
    dynamics = PendulumDynamics(dt)
    terminal_cost = PendulumTerminalCost()
    stage_cost = PendulumStageCost(dt, gamma)
    ocp = maafa.ocp.OCP(dynamics, stage_cost, terminal_cost, N)

    x = torch.rand(N+1, nbatch, dynamics.dimx)
    u = torch.rand(N, nbatch, dynamics.dimu)
    lmd = torch.rand(N+1, nbatch, dynamics.dimx)
    x0 = torch.rand(dynamics.dimx)

    x, u, lmd = ocp.solve(x0, x, u, lmd, verbose=True)
