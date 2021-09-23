from functools import partial
import torch
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import numpy as np

import maafa


class PendulumDynamics(maafa.dynamics.Dynamics):
    def __init__(self, dt=0.05):
        super().__init__()
        self.dimx = 2
        self.dimu = 1
        self.dt = dt
        # gravity (g), mass (m), length (l)
        self.params = Variable(torch.Tensor((10., 1., 1.)))

    def eval(self, x, u):
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

        g, m, l = torch.unbind(self.params)
        th = x[:, 0].view(-1, 1)
        dth = x[:, 1].view(-1, 1)
        ddth = -3.*g/(2.*l)*torch.sin(th+np.pi) + 3.*u/(m*l**2.)
        th_res = th + self.dt * dth 
        dth_res = dth + self.dt * ddth 
        return torch.stack([th_res, dth_res]).transpose(1, 0).squeeze(-1)


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
        dth_res_partial_dth = (-self.dt*3.*g/(2.*l)*torch.cos(th+np.pi)).squeeze(-1)
        dth_res_partial_u = 3./(m*l**2.)*torch.ones(nbatch)
        partial_th = torch.stack([th_res_partial_th, dth_res_partial_th]).transpose(1, 0)
        partial_dth = torch.stack([th_res_partial_dth, dth_res_partial_dth]).transpose(1, 0)
        partial_u = torch.stack([th_res_partial_u, dth_res_partial_u]).transpose(1, 0)
        return torch.stack([partial_th, partial_dth, partial_u]).transpose(0, 1).transpose(1, 2)