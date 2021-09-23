import torch
from torch import nn

import numpy as np

from .riccati_recursion import RiccatiRecursion
from .kkt import KKT
from . import utils


class MPC(nn.Module):
    def __init__(self, dynamics, stage_cost, terminal_cost, N, GaussNewton=True):
        super().__init__()
        self.dynamics = dynamics
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.dimx = dynamics.dimx
        self.dimu = dynamics.dimu
        self.N = N
        self.GaussNewton = GaussNewton
        self.riccati_recursion = RiccatiRecursion(dynamics, N)

    def eval_kkt(self, x0, x, u, lmd):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            lmd = lmd.unsqueeze(0)
        N = self.N
        x0res = x[0] - x0 
        l = []
        lxu = []
        Q = []
        xres = []
        F = []
        for i in range(N):
            l.append(self.stage_cost.eval(x[i], u[i], i))
            lxu.append(self.stage_cost.eval_sens(x[i], u[i], i))
            Q.append(self.stage_cost.eval_hess(x[i], u[i], i))
            xres.append(self.dynamics.eval(x[i], u[i])-x[i+1])
            F.append(self.dynamics.eval_sens(x[i], u[i]))  
            lxu[-1] += utils.bmv(F[-1].transpose(1, 2), lmd[i+1]) 
            lxu[-1][:, :self.dimx] -= lmd[i]
            if not self.GaussNewton:
                Q[-1] += self.dynamics.eval_hess(x[i], u[i], lmd[i+1]) 
        l.append(self.terminal_cost.eval(x[N])) 
        lxu.append(self.terminal_cost.eval_sens(x[N])) 
        lxu[-1] -= lmd[N]
        Q.append(self.terminal_cost.eval_hess(x[N]))
        return KKT(l, lxu, Q, x0res, xres, F)

    def solve(self, x0, x, u, lmd, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        kkt = self.eval_kkt(x0, x, u, lmd)
        kkt_error = kkt.kkt_error()
        if verbose:
            print('Initial KKT error = ' + str(kkt_error))
        for i in range(iter_max):
            if torch.max(kkt_error) < kkt_tol:
                return x, u, lmd
            else:
                dx, du, dlmd = self.riccati_recursion.riccati_recursion(kkt)
                self.update_solution(x, u, lmd, dx, du, dlmd)
                kkt = self.eval_kkt(x0, x, u, lmd)
                kkt_error = kkt.kkt_error()
            if verbose:
                print('KKT error at ' + str(i) + 'th iter = ' + str(kkt_error))
        return x, u, lmd

    def update_solution(self, x, u, lmd, dx, du, dlmd):
        x += dx
        u += du
        lmd += dlmd