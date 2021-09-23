import torch

import math


class KKT(object):
    def __init__(self, l, lxu, Q, x0res, xres, F):
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

    def kkt_error(self):
        norm = torch.norm(self.x0res, dim=1)
        for i in range(len(self.xres)):
            norm += torch.norm(self.xres[i], dim=1)
        for i in range(len(self.lxu)):
            norm += torch.norm(self.lxu[i], dim=1)
        return norm