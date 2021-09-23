import torch

import math


class KKT(object):
    def __init__(self, N, l, lxu, Q, x0res, xres, F):
        super().__init__()
        self.nbatch = l.shape[0]
        assert l.dim() == 2
        assert lxu.dim() == 3
        assert Q.dim() == 4
        assert xres.dim() == 3
        assert F.dim() == 4
        self.N = N
        self.l = l
        self.lxu = lxu
        self.Q = Q
        self.x0res = x0res
        self.xres = xres
        self.F = F

    def get_stage_kkt_block(self, stage):
        nx = self.nx
        if stage == self.N:
            QxxN = self.Q[stage, :, :nx, :nx]
            lxN = self.lxu[stage, :, :nx]
            return QxxN, lxN
        else:
            Q = self.Q[stage, :, :]
            lxu = self.lxu[stage, :]
            F = self.F[stage, :, :]
            xres = self.xres[stage, :]
            return Q, lxu, F, xres

    def get_stage_kkt(self, stage):
        nx = self.nx
        if stage == self.N:
            QxxN = self.Q[stage, :, :nx, :nx]
            lxN = self.lxu[stage, :, :nx]
            return QxxN, lxN
        else:
            Qxx = self.Q[stage, :, :nx, :nx]
            Qxu = self.Q[stage, :, :nx, nx:]
            Quu = self.Q[stage, :, nx:, nx:]
            lx = self.lxu[stage, :, :nx]
            lu = self.lxu[stage, :, nx:]
            A = self.F[stage, :, :, :nx]
            B = self.F[stage, :, :, nx:]
            xres = self.xres[stage, :, :]
            return Qxx, Qxu, Quu, lx, lu, A, B, xres

    def get_stage_lin_dynamics(self, stage):
        nx = self.nx
        A = self.F[stage, :, :, :nx]
        B = self.F[stage, :, :, nx:]
        xres = self.xres[stage, :, :]
        return A, B, xres

    def kkt_error(self):
        norm1 = torch.norm(self.x0res)
        norm2 = torch.norm(self.xres)
        norm3 = torch.norm(self.lxu)
        return math.sqrt(norm1+norm2+norm3) 