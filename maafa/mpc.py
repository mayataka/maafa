import torch
from torch import nn

from . import ocp


class MPC(nn.Module):
    def __init__(self, dynamics, stage_cost, terminal_cost, N, GaussNewton=True, nbatch=1):
        super().__init__()
        self.ocp = ocp.OCP(dynamics, stage_cost, terminal_cost, N, GaussNewton)
        self.x = torch.zeros(N+1, nbatch, dynamics.dimx)
        self.u = torch.zeros(N, nbatch, dynamics.dimu)
        self.lmd = torch.zeros(N+1, nbatch, dynamics.dimx)
        self.nbatch = nbatch

    def mpc_step(self, x0, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        self.x, self.u, self.lmd = self.ocp.solve(x0, self.x, self.u, self.lmd, 
                                                  kkt_tol, iter_max, verbose)
        return self.u[0]
