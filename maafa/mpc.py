import torch
from torch import nn

from . import ocp


class MPC(nn.Module):
    def __init__(self, dynamics, stage_cost, terminal_cost, N, GaussNewton=True, 
                 nbatch=1, device=None):
        super().__init__()
        assert nbatch >= 1
        self.ocp = ocp.OCP(dynamics, stage_cost, terminal_cost, N, GaussNewton)
        self.x = torch.zeros(N+1, nbatch, dynamics.dimx, device=device)
        self.u = torch.zeros(N, nbatch, dynamics.dimu, device=device)
        self.lmd = torch.zeros(N+1, nbatch, dynamics.dimx, device=device)
        self.gmm = torch.zeros(nbatch, dynamics.dimu, device=device)
        self.nbatch = nbatch
        self.device = device

    def reset_solution(self):
        self.x = torch.zeros(self.ocp.N+1, self.nbatch, self.ocp.dynamics.dimx, 
                             device=self.device)
        self.u = torch.zeros(self.ocp.N, self.nbatch, self.ocp.dynamics.dimu, 
                             device=self.device)
        self.lmd = torch.zeros(self.ocp.N+1, self.nbatch, self.ocp.dynamics.dimx, 
                               device=self.device)
        self.gmm = torch.zeros(self.nbatch, self.ocp.dynamics.dimu, 
                               device=self.device)

    def set_nbatch(self, nbatch):
        if not nbatch == self.nbatch:
            self.nbatch = nbatch
            self.reset_solution()

    def set_params(self, params):
        self.ocp.set_params(params)

    def check_params(self, eps=1.0e-06):
        self.ocp.check_params(eps)

    def mpc_step(self, x0, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        self.x, self.u, self.lmd = self.ocp.solve(x0=x0, x=self.x, u=self.u, 
                                                  lmd=self.lmd, kkt_tol=kkt_tol, 
                                                  iter_max=iter_max, 
                                                  verbose=verbose)
        return self.u[0].detach()


    def Q_step(self, x0, u0, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        assert x0.dim() == u0.dim()
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
            u0 = u0.unsqueeze(0)
        assert x0.dim() == 2
        assert x0.shape[0] == u0.shape[0]
        self.x, self.u, self.lmd, self.gmm, = self.ocp.Q_solve(x0, u0, self.x, 
                                                               self.u, self.lmd, 
                                                               self.gmm, 
                                                               kkt_tol=kkt_tol, 
                                                               iter_max=iter_max, 
                                                               verbose=verbose)


    def forward(self, x0, u0=None):
        return self.ocp.forward(x0, self.x, self.u, self.lmd, u0=u0, gmm=self.gmm)