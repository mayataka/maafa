import torch
from torch.autograd import Variable 

from maafa import utils


class QuadraticTerminalCost(torch.nn.Module):
    def __init__(self, params):
        super(QuadraticTerminalCost, self).__init__()
        self.default_xfref = utils.get_data(params.xfref).detach().clone()
        self.default_Vf_hess = utils.get_data(params.Vf_hess).detach().clone()
        self.default_Vf_grad = utils.get_data(params.Vf_grad).detach().clone()
        self.default_Vf_const = utils.get_data(params.Vf_const).detach().clone()
        self.xfref = params.xfref
        self.Vf_hess = params.Vf_hess
        self.Vf_grad = params.Vf_grad
        self.Vf_const = params.Vf_const

    def set_params(self, params):
        if params is not None:
            if  params.xfref is not None:
                self.xfref = params.xfref
            if params.Vf_hess is not None:
                self.Vf_hess = params.Vf_hess
            if params.Vf_grad is not None:
                self.Vf_grad = params.Vf_grad
            if params.Vf_const is not None:
                self.Vf_const = params.Vf_const

    def check_hessian(self, eps=1.0e-06):
        self.Vf_hess.data = utils.symmetrize(self.Vf_hess.data)
        self.Vf_hess.data = utils.make_positive_definite(self.Vf_hess.data, eps)

    def eval(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2
        if x.is_cuda and not self.xfref.is_cuda:
            self.xfref = self.xfref.cuda()
            self.Vf_hess = self.Vf_hess.cuda()
        xfref = self.xfref.clone()
        Vf_hess = self.Vf_hess.clone()
        Vf_hess = utils.symmetrize(Vf_hess)
        xdiff = x - xfref
        Wxdiff = Vf_hess.mm(xdiff.transpose(0, 1)).transpose(1, 0)
        Vf = torch.stack([xdiff[i].dot(Wxdiff[i]) for i in range(x.shape[0])])
        Vf_grad = self.Vf_grad.clone()
        Vf = Vf + torch.stack([Vf_grad.dot(x[i]) for i in range(x.shape[0])])
        Vf_const = self.Vf_const.clone()
        Vf = Vf + Vf_const
        return Vf

    def eval_sens(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2
        if x.is_cuda and not self.xfref.is_cuda:
            self.xfref = self.xfref.cuda()
            self.Vf_hess = self.Vf_hess.cuda()
        xfref = self.xfref.clone()
        Vf_hess = self.Vf_hess.clone()
        Vf_hess = utils.symmetrize(Vf_hess)
        xdiff = x - xfref
        Vfx = Vf_hess.mm(xdiff.transpose(0, 1)).transpose(1, 0)
        Vf_grad = self.Vf_grad.clone()
        Vfx = Vfx + Vf_grad
        return Vfx

    def eval_hess(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2
        if x.is_cuda and not self.xfref.is_cuda:
            self.xfref = self.xfref.cuda()
            self.Vf_hess = self.Vf_hess.cuda()
        Vf_hess = self.Vf_hess.clone()
        Vf_hess = utils.symmetrize(Vf_hess)
        Vfxx = torch.stack([Vf_hess for i in range (x.shape[0])]) 
        return Vfxx

    def forward(self, x):
        return self.eval(x)


class QuadraticStageCost(torch.nn.Module):
    def __init__(self, dt, gamma, params):
        super(QuadraticStageCost, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.default_xuref = utils.get_data(params.xuref).detach().clone()
        self.default_L_hess = utils.get_data(params.L_hess).detach().clone()
        self.default_L_grad = utils.get_data(params.L_grad).detach().clone()
        self.default_L_const = utils.get_data(params.L_const).detach().clone()
        self.xuref_true = self.default_xuref.detach().clone()
        self.L_hess_true = self.default_L_hess.detach().clone()
        self.xuref = params.xuref
        self.L_hess = params.L_hess
        self.L_grad = params.L_grad
        self.L_const = params.L_const

    def set_params(self, params):
        if params is not None and params.xuref is not None:
            self.xuref = params.xuref
        if params is not None and params.L_hess is not None:
            self.L_hess = params.L_hess
        if params.L_grad is not None:
            self.L_grad = params.L_grad
        if params.L_const is not None:
            self.L_const = params.L_const

    def check_hessian(self, eps=1.0e-06):
        self.L_hess.data = utils.symmetrize(self.L_hess.data)
        self.L_hess.data = utils.make_positive_definite(self.L_hess.data, eps)

    def eval_true(self, x, u):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert u.dim() == 2
        if x.is_cuda and not self.xuref_true.is_cuda:
            self.xuref_true = self.xuref_true.cuda()
            self.L_hess_true = self.L_hess_true.cuda()
        xuref = self.xuref_true.clone()
        L_hess = self.L_hess_true.clone()
        L_hess = utils.symmetrize(L_hess)
        xu = torch.cat([x.transpose(0, 1), u.transpose(0, 1)]).transpose(1, 0)
        xudiff = xu - xuref
        Wxudiff = L_hess.mm(xudiff.transpose(0, 1)).transpose(1, 0)
        return self.dt * 0.5 * torch.stack([xudiff[i].dot(Wxudiff[i]) for i in range(x.shape[0])])

    def eval(self, x, u, stage):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert u.dim() == 2
        assert stage >= 0
        if x.is_cuda and not self.xuref.is_cuda:
            self.xuref = self.xuref.cuda()
            self.L_hess = self.L_hess.cuda()
        xuref = self.xuref.clone()
        L_hess = self.L_hess.clone()
        L_hess = utils.symmetrize(L_hess)
        xu = torch.cat([x.transpose(0, 1), u.transpose(0, 1)]).transpose(1, 0)
        xudiff = xu - xuref
        Wxudiff = L_hess.mm(xudiff.transpose(0, 1)).transpose(1, 0)
        discount = self.gamma**stage
        L = discount * self.dt * 0.5 * torch.stack([xudiff[i].dot(Wxudiff[i]) for i in range(x.shape[0])])
        L_grad = self.L_grad.clone()
        L = L + discount * self.dt * torch.stack([L_grad.dot(xu[i]) for i in range(xu.shape[0])])
        L_const = self.L_const.clone()
        L = L + discount * self.dt * L_const
        return L

    def eval_sens(self, x, u, stage):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert u.dim() == 2
        if x.is_cuda and not self.xuref.is_cuda:
            self.xuref = self.xuref.cuda()
            self.L_hess = self.L_hess.cuda()
        xuref = self.xuref.clone()
        L_hess = self.L_hess.clone()
        L_hess = utils.symmetrize(L_hess)
        xu = torch.cat([x.transpose(0, 1), u.transpose(0, 1)]).transpose(1, 0)
        xudiff = xu - xuref
        discount = self.gamma**stage
        Lxu = discount * self.dt * L_hess.mm(xudiff.transpose(0, 1)).transpose(1, 0)
        L_grad = self.L_grad.clone()
        Lxu = Lxu + discount * self.dt * L_grad
        return Lxu

    def eval_hess(self, x, u, stage):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert u.dim() == 2
        if x.is_cuda and not self.xuref.is_cuda:
            self.xuref = self.xuref.cuda()
            self.L_hess = self.L_hess.cuda()
        L_hess = self.L_hess.clone()
        L_hess = utils.symmetrize(L_hess)
        discount = self.gamma**stage
        Qxu = discount * self.dt * torch.stack([L_hess for i in range (x.shape[0])]) 
        return Qxu

    def forward(self, x, u, stage):
        return self.eval(x, u, stage)