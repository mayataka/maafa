import torch
from torch import nn

import numpy as np
from torch._C import Value

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

    def set_params(self, params):
        self.dynamics.set_params(params)
        self.stage_cost.set_params(params)
        self.terminal_cost.set_params(params)

    def eval_kkt(self, x0, x, u, lmd, params=None):
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
            l.append(self.stage_cost.eval(x[i], u[i], i, params))
            lxu.append(self.stage_cost.eval_sens(x[i], u[i], i, params))
            Q.append(self.stage_cost.eval_hess(x[i], u[i], i, params))
            xres.append(self.dynamics.eval(x[i], u[i], params)-x[i+1])
            F.append(self.dynamics.eval_sens(x[i], u[i], params))  
            lxu[-1] = lxu[-1] + utils.bmv(F[-1].transpose(1, 2), lmd[i+1]) 
            lxu[-1][:, :self.dimx] = lxu[-1][:, :self.dimx] - lmd[i]
            if not self.GaussNewton:
                Q[-1] = Q[-1] + self.dynamics.eval_hess(x[i], u[i], lmd[i+1], params) 
        l.append(self.terminal_cost.eval(x[N], params)) 
        lxu.append(self.terminal_cost.eval_sens(x[N], params)) 
        lxu[-1] = lxu[-1] - lmd[N]
        Q.append(self.terminal_cost.eval_hess(x[N], params))
        return KKT(l, lxu, Q, x0res, xres, F)

    def solve(self, x0, x, u, lmd, params=None, kkt_tol=1.0e-04, iter_max=100, verbose=False):
        kkt = self.eval_kkt(x0, x, u, lmd, params)
        kkt_error = kkt.get_kkt_error()
        if verbose:
            print('Initial KKT error = ' + str(kkt_error))
        for i in range(iter_max):
            if torch.max(kkt_error) < kkt_tol:
                V_fn = kkt.get_lagrangian(lmd)
                return x, u, lmd, V_fn
            else:
                dx, du, dlmd = self.riccati_recursion.riccati_recursion(kkt)
                x = x + dx
                u = u + du
                lmd = lmd + dlmd
                kkt = self.eval_kkt(x0, x, u, lmd, params)
                kkt_error = kkt.get_kkt_error()
            if verbose:
                print('KKT error at ' + str(i+1) + 'th iter = ' + str(kkt_error))
        V_fn = kkt.get_lagrangian(lmd)
        return x, u, lmd, V_fn

    def eval_Q_kkt(self, x0, u0, x, u, lmd, gmm, params=None):
        kkt = self.eval_kkt(x0, x, u, lmd, params)
        u0res = u[0] - u0
        kkt.lxu[0][:, self.dimx:] = kkt.lxu[0][:, self.dimx:] + gmm
        return KKT(kkt.l, kkt.lxu, kkt.Q, kkt.x0res, kkt.xres, kkt.F, u0res)

    def Q_solve(self, x0, u0, x, u, lmd, gmm, params, 
                kkt_tol=1.0e-04, iter_max=100, verbose=False):
        kkt = self.eval_Q_kkt(x0, u0, x, u, lmd, gmm, params)
        kkt_error = kkt.get_Q_kkt_error()
        if verbose:
            print('Initial Q-KKT error = ' + str(kkt_error))
        for i in range(iter_max):
            if torch.max(kkt_error) < kkt_tol:
                Q_fn = kkt.get_Q_function(lmd, gmm)
                return x, u, lmd, gmm, Q_fn
            else:
                dx, du, dlmd, dgmm = self.riccati_recursion.Q_riccati_recursion(kkt)
                x = x + dx
                u = u + du
                lmd = lmd + dlmd
                gmm = gmm + dgmm
                kkt = self.eval_Q_kkt(x0, u0, x, u, lmd, gmm, params)
                kkt_error = kkt.get_Q_kkt_error()
            if verbose:
                print('Q-KKT error at ' + str(i+1) + 'th iter = ' + str(kkt_error))
        Q_fn = kkt.get_Q_function(lmd, gmm)
        return x, u, lmd, gmm, Q_fn

    # Comptues Q function 
    def forward(self, x0, u0, x, u, lmd, gmm, params):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            lmd = lmd.unsqueeze(0)
        N = self.N
        x0res = x[0] - x0 
        u0res = u[0] - u0
        l = []
        xres = []
        for i in range(N):
            l.append(self.stage_cost.forward(x[i], u[i], i, params))
            xres.append(self.dynamics.forward(x[i], u[i], params)-x[i+1])
        l.append(self.terminal_cost.forward(x[N], params)) 
        kkt = KKT(l=l, lxu=None, Q=None, x0res=x0res, xres=xres, F=None, u0res=u0res)
        Q_fn = kkt.get_Q_function(lmd, gmm)
        return Q_fn