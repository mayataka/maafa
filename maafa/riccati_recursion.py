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
            Q, lxu, F, xres = kkt.get_stage_kkt_block(i)
            Ft = F.transpose(1, 2)
            print("F: ", F)
            print("Ft: ", Ft)
            print("Q: ", Q)
            print("P[-1]: ", P[-1])
            FHG = Q + Ft.bmm(P[-1]).bmm(F)
            Fi = FHG[:, :dimx, :dimx]
            Hi = FHG[:, :dimx, dimx:]
            Gi = FHG[:, dimx:, dimx:]
            Ginv = torch.stack([torch.pinverse(Gi[j]) for j in range(Gi.shape[0])])
            Ki = -Ginv.bmm(Hi.transpose(1, 2))
            vi = utils.bmv(Ft, (utils.bmv(P[-1], xres)-s[-1])) + lxu
            vxi = vi[:, :dimx]
            vui = vi[:, dimx:]
            ki = utils.bmv(-Ginv, vui)
            Pi = Fi - Ki.transpose(1, 2).bmm(Gi).bmm(Ki)
            si = - vxi - utils.bmv(Hi, ki)
            P.append(Pi)
            s.append(si)
            K.append(Ki)
            k.append(ki)
        P.reverse()
        s.reverse()
        K.reverse()
        k.reverse()
        P = torch.stack(P)
        s = torch.stack(s)
        K = torch.stack(K)
        k = torch.stack(k)
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