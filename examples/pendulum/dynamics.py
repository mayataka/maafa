import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from maafa.dynamics import Dynamics

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

class PendulumDynamics(Dynamics):
    def __init__(self, dt=0.05):
        super().__init__()
        self.nx = 2
        self.nu = 1
        self.dt = dt
        # gravity (g), mass (m), length (l)
        self.params = Variable(torch.Tensor((10., 1., 1.)))

    @classmethod
    def eval(self, x, u, x1):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            x1 = x1.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        assert x1.dim() == 2
        assert x1.shape[0] == u.shape[0]
        assert x1.shape[1] == 2

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        g, m, l = torch.unbind(self.params)
        th = x[:, 0].view(-1, 1)
        dth = x[:, 1].view(-1, 1)
        th1 = x1[:, 0].view(-1, 1)
        dth1 = x1[:, 1].view(-1, 1)
        ddth = -3.*g/(2.*l)*torch.sin(th+np.pi) + 3.*u/(m*l**2.)
        th_res = th + self.dt * dth - th1
        dth_res = dth + self.dt * ddth - dth1
        return torch.Tensor([th_res, dth_res])

    @classmethod
    def eval_sens(self, x, u):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        nbatch = x.shape[0]

        g, m, l = torch.unbind(self.params)
        th = x[:, 0].view(-1, 1)
        th_res_partial_th = torch.ones(nbatch)
        th_res_partial_dth = self.dt*torch.ones(nbatch) 
        th_res_partial_u = torch.zeros(nbatch)
        dth_res_partial_th = torch.ones(nbatch)
        dth_res_partial_dth = -self.dt*3.*g/(2.*l)*torch.cos(th+np.pi)
        dth_res_partial_u = 3./(m*l**2.)*torch.ones(nbatch)
        th_res_partial = torch.Tensor([th_res_partial_th, th_res_partial_dth, th_res_partial_u])
        dth_res_partial = torch.Tensor([dth_res_partial_th, dth_res_partial_dth, dth_res_partial_u])
        return torch.Tensor([th_res_partial, dth_res_partial])