import torch


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
        self.dimx = xres[0].shape[1]
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
            norm = norm + torch.norm(self.xres[i], dim=1)
        for i in range(len(self.lxu)):
            norm = norm + torch.norm(self.lxu[i], dim=1)
        return norm

    def get_Q_kkt_error(self):
        norm = self.get_kkt_error()
        if self.u0res is not None:
            norm = norm + torch.norm(self.u0res, dim=1)
        return norm

    def get_lagrangian(self, lmd):
        nbatch = lmd.shape[1]
        lag = self.l[self.N]
        for i in range(self.N):
            lag = lag + self.l[i]
            lag = lag + torch.stack([lmd[i+1, j].dot(self.xres[i][j]) for j in range(nbatch)])
        lag = lag - torch.stack([lmd[0, j].dot(self.x0res[j]) for j in range(nbatch)])
        return lag

    def get_Q_function(self, lmd, gmm):
        lag = self.get_lagrangian(lmd)
        nbatch = gmm.shape[0]
        lag = lag + torch.stack([gmm[j].dot(self.u0res[j]) for j in range(nbatch)])
        return lag
