import torch
from torch.autograd import Variable
import numpy as np

from matplotlib import pyplot as plt


class PendulumDynamics(torch.nn.Module):
    def __init__(self, dt=0.05, params=None):
        super(PendulumDynamics, self).__init__()
        self.dimx = 2
        self.dimu = 1
        self.dt = dt
        # gravity (g), mass (m), length (l)
        self.default_params = Variable(torch.Tensor((10., 1., 1.)))
        if params is not None and params.dyn_params is not None:
            self.params = params.dyn_params
        else:
            self.params = self.default_params

    def set_params(self, params):
        if params is not None and params.dyn_params is not None:
            self.params = params.dyn_params

    def eval(self, x, u, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        self.set_params(params)
        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        g, m, l = torch.unbind(self.params)
        g = g.clone()
        m = m.clone()
        l = l.clone()
        th = x[:, 0].view(-1, 1)
        dth = x[:, 1].view(-1, 1)
        ddth = - 3.*g/(2.*l)*torch.sin(th.clone()+np.pi) + 3.*u.clone()/(m*l**2.)
        th_res = th + self.dt * dth 
        dth_res = dth + self.dt * ddth 
        return torch.stack([th_res, dth_res]).transpose(1, 0).squeeze(-1)

    def eval_sens(self, x, u, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        self.set_params(params)
        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()
        nbatch = x.shape[0]
        g, m, l = torch.unbind(self.params)
        g = g.clone()
        m = m.clone()
        l = l.clone()
        th = x[:, 0].view(-1, 1)
        device = x.device
        th_res_partial_th = torch.ones(nbatch, device=device)
        th_res_partial_dth = self.dt * torch.ones(nbatch, device=device) 
        th_res_partial_u = torch.zeros(nbatch, device=device)
        dth_res_partial_th = self.dt * (-3.*g/(2.*l)*torch.cos(th.clone()+np.pi)).squeeze(-1)
        dth_res_partial_dth = torch.ones(nbatch, device=device)
        dth_res_partial_u = (self.dt*3./(m*l**2.))*torch.ones(nbatch, device=device)
        partial_th = torch.stack([th_res_partial_th, dth_res_partial_th]).transpose(1, 0)
        partial_dth = torch.stack([th_res_partial_dth, dth_res_partial_dth]).transpose(1, 0)
        partial_u = torch.stack([th_res_partial_u, dth_res_partial_u]).transpose(1, 0)
        return torch.stack([partial_th, partial_dth, partial_u]).transpose(0, 1).transpose(1, 2)

    def eval_hess(self, x, u, parms=None):
        return NotImplementedError()

    def forward(self, x, u, params):
        return self.eval(x, u, params)

    def reset(self, nbatch=1, device=None):
        return np.pi*torch.rand(nbatch, self.dimx, device=device)

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