from matplotlib.pyplot import disconnect
import torch
from torch import nn

import numpy as np

from .riccati_recursion import RiccatiRecursion
from .kkt import KKT
from . import utils


class OCP(nn.Module):
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
            l.append(self.stage_cost.forward(x[i], u[i], i))
            lxu.append(self.stage_cost.eval_sens(x[i], u[i], i))
            Q.append(self.stage_cost.eval_hess(x[i], u[i], i))
            xres.append(self.dynamics.forward(x[i], u[i])-x[i+1])
            F.append(self.dynamics.eval_sens(x[i], u[i]))  
            lxu[-1] += utils.bmv(F[-1].transpose(1, 2), lmd[i+1]) 
            lxu[-1][:, :self.dimx] -= lmd[i]
            if not self.GaussNewton:
                Q[-1] += self.dynamics.eval_hess(x[i], u[i], lmd[i+1]) 
        l.append(self.terminal_cost.forward(x[N])) 
        lxu.append(self.terminal_cost.eval_sens(x[N])) 
        lxu[-1] -= lmd[N]
        Q.append(self.terminal_cost.eval_hess(x[N]))
        return KKT(l, lxu, Q, x0res, xres, F)

    def solve(self, x0, x, u, lmd, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        kkt = self.eval_kkt(x0, x, u, lmd)
        kkt_error = kkt.get_kkt_error()
        if verbose:
            print('Initial KKT error = ' + str(kkt_error))
        for i in range(iter_max):
            if torch.max(kkt_error) < kkt_tol:
                return x, u, lmd
            else:
                dx, du, dlmd = self.riccati_recursion.riccati_recursion(kkt)
                x += dx
                u += du
                lmd += dlmd
                kkt = self.eval_kkt(x0, x, u, lmd)
                kkt_error = kkt.get_kkt_error()
            if verbose:
                print('KKT error at ' + str(i) + 'th iter = ' + str(kkt_error))
        return x, u, lmd

    def eval_Q_kkt(self, x0, u0, x, u, lmd, gmm):
        kkt = self.eval_kkt(x0, x, u, lmd)
        u0res = u[0] - u0
        kkt.lxu[0][:, self.dimx:] += gmm
        return KKT(kkt.l, kkt.lxu, kkt.Q, kkt.x0res, kkt.xres, kkt.F, u0res)

    def Q_solve(self, x0, u0, x, u, lmd, gmm, kkt_tol=1.0e-04, 
                iter_max=100, verbose=False):
        kkt = self.eval_Q_kkt(x0, u0, x, u, lmd, gmm)
        kkt_error = kkt.get_Q_kkt_error()
        if verbose:
            print('Initial Q-KKT error = ' + str(kkt_error))
        for i in range(iter_max):
            if torch.max(kkt_error) < kkt_tol:
                return x, u, lmd, gmm
            else:
                dx, du, dlmd, dgmm = self.riccati_recursion.Q_riccati_recursion(kkt)
                x += dx
                u += du
                lmd += dlmd
                gmm += dgmm
                kkt = self.eval_Q_kkt(x0, u0, x, u, lmd, gmm)
                kkt_error = kkt.get_Q_kkt_error()
            if verbose:
                print('Q-KKT error at ' + str(i) + 'th iter = ' + str(kkt_error))
        return x, u, lmd, gmm

    # Comptues Q function and returns TD error
    def forward(self, x0, x1, u0, x, u, lmd, gmm, kkt_tol=1.0e-04, 
                iter_max=100, verbose=False):
        L0 = self.stage_cost.forward(x0, u0, stage=0)
        x, u, lmd, gmm = self.Q_solve(x0, u0, x, u, lmd, gmm, 
                                      kkt_tol, iter_max, verbose)
        kkt = self.eval_Q_kkt(x0, u0, x, u, lmd, gmm)
        Q0 = kkt.get_Q_function(lmd, gmm)
        x, u, lmd = self.solve(x1, x, u, lmd, kkt_tol, iter_max, verbose)
        kkt = self.eval_kkt(x1, x, u, lmd)
        V1 = kkt.get_lagrangian(lmd)
        discount = self.stage_cost.gamma
        return L0 + discount*V1 - Q0 