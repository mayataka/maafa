import torch
from torch import nn

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

    def check_params(self, eps=1.0e-06):
        self.stage_cost.check_hessian(eps)
        self.terminal_cost.check_hessian(eps)

    def eval_kkt(self, x0, x, u, lmd):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            lmd = lmd.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            lmd = lmd.unsqueeze(0)
        assert x0.dim() == 2
        assert x.dim() == 3
        assert u.dim() == 3
        assert lmd.dim() == 3
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
            lxu[-1] = lxu[-1] + utils.bmv(F[-1].transpose(1, 2), lmd[i+1]) 
            lxu[-1][:, :self.dimx] = lxu[-1][:, :self.dimx] - lmd[i]
            if not self.GaussNewton:
                Q[-1] = Q[-1] + self.dynamics.eval_hess(x[i], u[i], lmd[i+1]) 
        l.append(self.terminal_cost.eval(x[N])) 
        lxu.append(self.terminal_cost.eval_sens(x[N])) 
        lxu[-1] = lxu[-1] - lmd[N]
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
                x = x + dx
                u = u + du
                lmd = lmd + dlmd
                kkt = self.eval_kkt(x0, x, u, lmd)
                kkt_error = kkt.get_kkt_error()
            if verbose:
                print('KKT error at ' + str(i+1) + 'th iter = ' + str(kkt_error))
        return x, u, lmd

    def eval_Q_kkt(self, x0, u0, x, u, lmd, gmm):
        assert u0.dim() == 2
        assert gmm.dim() == 2
        kkt = self.eval_kkt(x0, x, u, lmd)
        u0res = u[0] - u0
        kkt.lxu[0][:, self.dimx:] = kkt.lxu[0][:, self.dimx:] + gmm
        return KKT(kkt.l, kkt.lxu, kkt.Q, kkt.x0res, kkt.xres, kkt.F, u0res)

    def Q_solve(self, x0, u0, x, u, lmd, gmm, 
                kkt_tol=1.0e-04, iter_max=100, verbose=False):
        kkt = self.eval_Q_kkt(x0, u0, x, u, lmd, gmm)
        kkt_error = kkt.get_Q_kkt_error()
        if verbose:
            print('Initial Q-KKT error = ' + str(kkt_error))
        for i in range(iter_max):
            if torch.max(kkt_error) < kkt_tol:
                return x, u, lmd, gmm
            else:
                dx, du, dlmd, dgmm = self.riccati_recursion.Q_riccati_recursion(kkt)
                x = x + dx
                u = u + du
                lmd = lmd + dlmd
                gmm = gmm + dgmm
                kkt = self.eval_Q_kkt(x0, u0, x, u, lmd, gmm)
                kkt_error = kkt.get_Q_kkt_error()
            if verbose:
                print('Q-KKT error at ' + str(i+1) + 'th iter = ' + str(kkt_error))
        return x, u, lmd, gmm

    # Comptues V or Q function 
    def forward(self, x0, x, u, lmd, u0=None, gmm=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            lmd = lmd.unsqueeze(0)
        N = self.N
        x0res = x[0] - x0 
        if u0 is not None:
            u0res = u[0] - u0
        l = []
        xres = []
        for i in range(N):
            l.append(self.stage_cost.forward(x[i], u[i], i))
            xres.append(self.dynamics.forward(x[i], u[i])-x[i+1])
        l.append(self.terminal_cost.forward(x[N])) 
        if u0 is not None:
            assert gmm is not None
            kkt = KKT(l=l, lxu=None, Q=None, x0res=x0res, xres=xres, F=None, u0res=u0res)
            Q_fn = kkt.get_Q_function(lmd, gmm)
            return Q_fn
        else:
            kkt = KKT(l=l, lxu=None, Q=None, x0res=x0res, xres=xres, F=None, u0res=None)
            V_fn = kkt.get_V_function(lmd)
            return V_fn