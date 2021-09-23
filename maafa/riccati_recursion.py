import torch
from . import utils


class RiccatiRecursion(object):
    def __init__(self, dynamics, N):
        super().__init__()
        self.nx = dynamics.nx
        self.nu = dynamics.nu
        self.N = N

    def riccati_recursion(self, kkt):
        P, s, K, k = self.backward_riccati_recursion(kkt)
        dx, du, dlmd = self.forward_riccati_recursion(kkt, P, s, K, k)
        return dx, du, dlmd

    def backward_riccati_recursion(self, kkt):
        nx = self.nx
        N = self.N
        P = []
        s = []
        K = []
        k = []
        QxxN, lxN = kkt.get_stage_kkt(N)
        P.append(QxxN)
        s.append(-lxN)
        assert QxxN.dim() == 3
        for i in range(N-1, -1, -1):
            Q, lxu, F, xres = kkt.get_stage_kkt_block(i)
            Ft = F.transpose(1, 2)
            FHG = Q + Ft.bmm(P[-1]).bmm(F)
            Fi = FHG[i, :, :nx, :nx]
            Hi = FHG[i, :, :nx, nx:]
            Gi = FHG[i, nx:, nx:]
            Ginv = [torch.pinverse(Gi) for j in range(Gi.shape[0])]
            Ginv = torch.stack(Ginv)
            Ki = -Ginv.bmm(Hi)
            vi = utils.bmv(Ft, (utils.bmv(P[-1], xres)-s[-1])) + lxu
            vxi = vi[:, :nx]
            vui = vi[:, nx:]
            ki = utils.bmv(-Ginv, vui)
            Pi = Fi - Ki.transpose(1, 2).bmm(Gi).bmm(Ki)
            si = -vxi - Hi.bmv(ki)
            P.append(Pi)
            s.append(si)
            K.append(Ki)
            k.append(ki)
        P.reverse()
        s.reverse()
        K.reverse()
        k.reverse()
        x0res = kkt.x0res
        return P, s, K, k

    def forward_riccati_recursion(self, kkt, P, s, K, k):
        N = self.N
        dx = []
        du = []
        dlmd = []
        dx.append(-kkt.x0res)
        for i in range(N):
            A, B, xres = kkt.get_stage_lin_dynamics(i)
            dui = utils.bmv(K[i], dx[-1]) + k[i]
            dlmdi = utils.bmv(P[i], dx[-1]) - self.s[i]
            dxip1 = utils.bmv(A, dx[-1]) + utils.bmv(B, du[i]) + xres
            dx.append(dxip1)
            du.append(dui)
            dlmd.append(dlmdi)
        dlmdN = utils.bmv(P[N], dx[-1]) - s[N]
        dlmd.append(dlmdN)
        dx = torch.Tensor(dx)
        du = torch.Tensor(du)
        dlmd = torch.Tensor(dlmd)
        return dx, du, dlmd