import torch
from . import utils


class RiccatiRecursion(object):
    def __init__(self, dynamics, N):
        super().__init__()
        self.dimx = dynamics.dimx
        self.dimu = dynamics.dimu
        self.N = N

    def riccati_recursion(self, kkt):
        P, s, K, k = self.backward_riccati_recursion(kkt)
        dx, du, dlmd = self.forward_riccati_recursion(kkt, P, s, K, k)
        return dx, du, dlmd

    def backward_riccati_recursion(self, kkt):
        dimx = self.dimx
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
            Q, lxu, F, xres = kkt.get_stage_kkt(i)
            Ft = F.transpose(1, 2)
            FHG = Q + Ft.bmm(P[-1]).bmm(F)
            Fi = FHG[:, :dimx, :dimx]
            Hi = FHG[:, :dimx, dimx:]
            Gi = FHG[:, dimx:, dimx:]
            Ginv = torch.stack([torch.pinverse(Gi[j]) for j in range(Gi.shape[0])])
            Ki = -Ginv.bmm(Hi.transpose(1, 2))
            vi = utils.bmv(Ft, (utils.bmv(P[-1], xres)-s[-1])) + lxu
            vxi = vi[:, :dimx]
            vui = vi[:, dimx:]
            Pi = Fi - Ki.transpose(1, 2).bmm(Gi).bmm(Ki)
            ki = - utils.bmv(Ginv, vui)
            si = - vxi - utils.bmv(Hi, ki)
            P.append(Pi)
            s.append(si)
            K.append(Ki)
            k.append(ki)
        P.reverse()
        s.reverse()
        K.reverse()
        k.reverse()
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
            dlmdi = utils.bmv(P[i], dx[-1]) - s[i]
            du.append(dui)
            dlmd.append(dlmdi)
            dxi = utils.bmv(A, dx[-1]) + utils.bmv(B, du[-1]) + xres
            dx.append(dxi)
        dlmdN = utils.bmv(P[N], dx[-1]) - s[N]
        dlmd.append(dlmdN)
        dx = torch.stack(dx)
        du = torch.stack(du)
        dlmd = torch.stack(dlmd)
        return dx, du, dlmd

    def Q_backward_riccati_recursion(self, kkt):
        P, s, K, k = self.backward_riccati_recursion(kkt)
        Q0, lxu0, F0, xres0 = kkt.get_stage_kkt(0)
        F0t = F0.transpose(1, 2)
        FHG0 = Q0 + F0t.bmm(P[1]).bmm(F0)
        dimx = self.dimx
        H0 = FHG0[:, :dimx, dimx:]
        G0 = FHG0[:, dimx:, dimx:]
        v0 = utils.bmv(F0t, (utils.bmv(P[1], xres0)-s[1])) + lxu0
        vu0 = v0[:, dimx:]
        return P, s, K, k, H0, G0, vu0

    def Q_forward_riccati_recursion(self, kkt, P, s, K, k, H0, G0, vu0):
        N = self.N
        dx = []
        du = []
        dlmd = []
        dx.append(-kkt.x0res)
        A0, B0, xres0 = kkt.get_stage_lin_dynamics(0)
        du0 = -kkt.u0res
        dgmm = - utils.bmv(H0.transpose(1, 2), dx[-1]) - utils.bmv(G0, du0) - vu0
        dlmd0 = utils.bmv(P[0], dx[-1]) - s[0]
        du.append(du0)
        dlmd.append(dlmd0)
        dx1 = utils.bmv(A0, dx[-1]) + utils.bmv(B0, du[-1]) + xres0
        dx.append(dx1)
        for i in range(1, N, 1):
            A, B, xres = kkt.get_stage_lin_dynamics(i)
            dui = utils.bmv(K[i], dx[-1]) + k[i]
            dlmdi = utils.bmv(P[i], dx[-1]) - s[i]
            du.append(dui)
            dlmd.append(dlmdi)
            dxi = utils.bmv(A, dx[-1]) + utils.bmv(B, du[-1]) + xres
            dx.append(dxi)
        dlmdN = utils.bmv(P[N], dx[-1]) - s[N]
        dlmd.append(dlmdN)
        dx = torch.stack(dx)
        du = torch.stack(du)
        dlmd = torch.stack(dlmd)
        return dx, du, dlmd, dgmm

    def Q_riccati_recursion(self, kkt):
        P, s, K, k, H0, G0, vu0 = self.Q_backward_riccati_recursion(kkt)
        dx, du, dlmd, dgmm = self.Q_forward_riccati_recursion(kkt, P, s, K, k, H0, G0, vu0)
        return dx, du, dlmd, dgmm