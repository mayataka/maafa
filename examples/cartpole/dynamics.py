import torch
from torch.nn.parameter import Parameter
import numpy as np

from matplotlib import pyplot as plt

from maafa import utils


class CartpoleDynamics(torch.nn.Module):
    def __init__(self, dt, params):
        super(CartpoleDynamics, self).__init__()
        assert dt > 0.
        self.dimx = 4
        self.dimu = 1
        self.dt = dt
        # gravity (g), mass of cart (M), mass of pole (m), length of pole (l)
        self.params = params.dyn_params
        self.bias = params.dyn_bias
        self.default_params = utils.get_data(params.dyn_params).detach().clone()
        self.default_bias = utils.get_data(params.dyn_bias).detach().clone()

    def set_params(self, params):
        if params is not None and params.dyn_params is not None:
            self.params = params.dyn_params
        if params is not None and params.dyn_bias is not None:
            self.bias = params.dyn_bias

    def eval(self, x, u):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 4
        assert u.shape[1] == 1
        assert u.dim() == 2
        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        g, M, m, l = torch.unbind(self.params)
        g = g.clone()
        M = M.clone()
        m = m.clone()
        l = l.clone()
        y, th, dy, dth = torch.unbind(x)
        cin = (u + m*l*dth**2*torch.sin(th)) / (M+m)
        ddth = (g*torch.sin(th)-torch.cos(th)*cin) / \
                 (l*(4./3.-m*torch.cos(th)**2 / (M+m)))
        ddy = cin - m*l*ddth*torch.cos(th) / (M+m)
        by, bth, bdy, bdth = torch.unbind(self.bias)
        by = by.clone()
        bth = bth.clone()
        bdy = bdy.clone()
        bdth = bdth.clone()
        y_res = y + self.dt * dy + by
        th_res = th + self.dt * dth + bth 
        dy_res = dy + self.dt * ddy + bdy
        dth_res = dth + self.dt * ddth + bdth 
        return torch.stack([y_res, th_res, dy_res, dth_res]).transpose(1, 0).squeeze(-1)

    def eval_sens(self, x, u):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 4
        assert u.shape[1] == 1
        assert u.dim() == 2
        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        nbatch = x.shape[0]
        g, M, m, l = torch.unbind(self.params)
        g = g.clone()
        M = M.clone()
        m = m.clone()
        l = l.clone()
        y, th, dy, dth = torch.unbind(x)

        cin = (u + m*l*dth**2*torch.sin(th)) / (M+m)
        ddth = (g*torch.sin(th)-torch.cos(th)*cin) / \
                 (l*(4./3.-m*torch.cos(th)**2 / (M+m)))
        ddy = cin - m*l*ddth*torch.cos(th) / (M+m)
        y_res = y + self.dt * dy + by
        th_res = th + self.dt * dth + bth 
        dy_res = dy + self.dt * ddy + bdy
        dth_res = dth + self.dt * ddth + bdth 

        device = x.device
        # y
        y_res_partial_y = torch.ones(nbatch, device=device)
        y_res_partial_th = torch.zeros(nbatch, device=device)
        y_res_partial_dy = self.dt * torch.ones(nbatch, device=device)
        y_res_partial_dth = torch.zeros(nbatch, device=device)
        y_res_partial_u = torch.zeros(nbatch, device=device)
        # th
        th_res_partial_y = torch.zeros(nbatch, device=device)
        th_res_partial_th = torch.ones(nbatch, device=device)
        th_res_partial_dy = torch.zeros(nbatch, device=device)
        th_res_partial_dth = self.dt * torch.ones(nbatch, device=device) 
        th_res_partial_u = torch.zeros(nbatch, device=device)
        # dy
        dy_res_partial_y = torch.zeros(nbatch, device=device)
        dy_res_partial_th = torch.zeros(nbatch, device=device)
        dy_res_partial_dy = self.dt * torch.ones(nbatch, device=device)
        dy_res_partial_dth = torch.zeros(nbatch, device=device)
        dy_res_partial_u = torch.zeros(nbatch, device=device)
        # dth
        dth_res_partial_y = torch.zeros(nbatch, device=device)
        dth_res_partial_th = torch.zeros(nbatch, device=device)
        dth_res_partial_dy = self.dt * torch.ones(nbatch, device=device)
        dth_res_partial_dth = torch.zeros(nbatch, device=device)
        dth_res_partial_u = torch.zeros(nbatch, device=device)

        # dth_res_partial_y = self.dt * (-3.*g/(2.*l)*torch.cos(th.clone()+np.pi)).squeeze(-1)
        # dth_res_partial_th = self.dt * (-3.*g/(2.*l)*torch.cos(th.clone()+np.pi)).squeeze(-1)
        # dth_res_partial_dth = torch.ones(nbatch, device=device)
        # dth_res_partial_u = (self.dt*3./(m*l**2.))*torch.ones(nbatch, device=device)
        partial_y = torch.stack([y_res_partial_y, th_res_partial_y, dy_res_partial_y, dth_res_partial_y]).transpose(1, 0)
        partial_th = torch.stack([y_res_partial_th, th_res_partial_th, dy_res_partial_th, dth_res_partial_th]).transpose(1, 0)
        partial_dy = torch.stack([y_res_partial_dy, th_res_partial_dy, dy_res_partial_dy, dth_res_partial_dy]).transpose(1, 0)
        partial_dth = torch.stack([y_res_partial_dth, th_res_partial_dth, dy_res_partial_dth, dth_res_partial_dth]).transpose(1, 0)
        partial_u = torch.stack([y_res_partial_u, th_res_partial_u, dy_res_partial_u, dth_res_partial_u]).transpose(1, 0)
        return torch.stack([partial_th, partial_dth, partial_u]).transpose(0, 1).transpose(1, 2)

    def eval_hess(self, x, u):
        return NotImplementedError()

    def forward(self, x, u):
        return self.eval(x, u)

    def reset(self, nbatch=1, device=None):
        return 2*np.pi*torch.rand(nbatch, self.dimx, device=device)-np.pi*torch.ones(nbatch, self.dimx, device=device)

    def get_frame(self, x, ax=None):
        x = x.view(-1)
        assert len(x) == 2
        th, dth = torch.unbind(x)
        g, m, l = torch.unbind(self.params)
        x = torch.sin(th)*l
        y = torch.cos(th)*l
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()
        ax.plot((0, x.detach().numpy()), (0, y.detach().numpy()), color='k')
        l = l.detach().numpy()
        ax.set_xlim((-l*1.2, l*1.2))
        ax.set_ylim((-l*1.2, l*1.2))
        return fig, ax