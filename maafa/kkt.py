import torch

import math


class KKT(object):
    def __init__(self, l, lxu, Q, x0res, xres, F, u0res=None):
        super().__init__()
        self.nbatch = l[0].shape[0]
        self.N = len(l)-1
        self.l = l
        self.lxu = lxu
        self.Q = Q
        self.x0res = x0res
        self.xres = xres
        self.F = F
        self.dimx = F[0].shape[1]
        self.u0res = u0res

    def get_stage_kkt(self, stage):
        if stage == self.N:
            return self.Q[stage], self.lxu[stage]
        else:
            return self.Q[stage], self.lxu[stage], self.F[stage], self.xres[stage]

    def get_stage_lin_dynamics(self, stage):
        dimx = self.dimx
        A = self.F[stage][:, :, :dimx]
        B = self.F[stage][:, :, dimx:]
        xres = self.xres[stage]
        return A, B, xres

    def get_kkt_error(self):
        norm = torch.norm(self.x0res, dim=1)
        for i in range(len(self.xres)):
            norm += torch.norm(self.xres[i], dim=1)
        for i in range(len(self.lxu)):
            norm += torch.norm(self.lxu[i], dim=1)
        return norm

    def get_Q_kkt_error(self):
        norm = self.get_kkt_error()
        if self.u0res is not None:
            norm += torch.norm(self.u0res, dim=1)
        return norm

    def get_lagrangian(self, lmd):
        lag = self.l[self.N]
        for i in range(self.N):
            lag += self.l[i]
            lag += lmd[i+1].dot(self.xres[i])
        lag += lmd[0].dot(self.x0res)
        return lag

    def get_Q_function(self, lmd, gmm):
        lag = self.get_lagrangian(lmd)
        lag += gmm.dot(self.u0res)
        return lag
