import torch
from torch import nn

from . import ocp


class MPC(nn.Module):
    def __init__(self, dynamics, stage_cost, terminal_cost, N, GaussNewton=True, 
                 nbatch=1, device=None):
        super().__init__()
        self.ocp = ocp.OCP(dynamics, stage_cost, terminal_cost, N, GaussNewton)
        self.x = torch.zeros(N+1, nbatch, dynamics.dimx, device=device)
        self.u = torch.zeros(N, nbatch, dynamics.dimu, device=device)
        self.lmd = torch.zeros(N+1, nbatch, dynamics.dimx, device=device)
        self.gmm = torch.zeros(nbatch, dynamics.dimu, device=device)
        self.nbatch = nbatch

    def set_params(self, params):
        self.ocp.set_params(params)

    def mpc_step(self, x0, params=None, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        self.x, self.u, self.lmd, V_fn = self.ocp.solve(x0=x0, x=self.x, 
                                                        u=self.u, lmd=self.lmd, 
                                                        params=params, 
                                                        kkt_tol=kkt_tol, 
                                                        iter_max=iter_max, 
                                                        verbose=verbose)
        return self.u[0].detach(), V_fn

    def Q_step(self, x0, u0, params=None, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        self.x, self.u, self.lmd, self.gmm, Q_fn = self.ocp.Q_solve(
            x0, u0, self.x, self.u, self.lmd, self.gmm, params=params, 
            kkt_tol=kkt_tol, iter_max=iter_max, verbose=verbose)

    def forward(self, x0, u0, params=None, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        return self.ocp.forward(x0, u0, self.x, self.u, self.lmd, self.gmm, 
                                params=params)