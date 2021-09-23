import torch
from torch.autograd import Function, Variable 

import maafa 
from maafa import utils


class PendulumTerminalCost(maafa.cost.TerminalCost):
    def __init__(self):
        super().__init__()
        self.xref_true = torch.Tensor([1., 0.])
        self.xweight_true = torch.Tensor([[1., 0.], [0., 1.,]])
        self.xref = Variable(self.xref_true) 
        self.xweight = Variable(self.xweight_true)

    def eval(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.is_cuda and not self.xref.is_cuda:
            self.xref = self.xref.cuda()
            self.xweight = self.xweight.cuda()
        xdiff = x - self.xref
        Wxdiff = self.xweight.mm(xdiff.transpose(0, 1)).transpose(1, 0)
        return 0.5 * torch.stack([xdiff[i].dot(Wxdiff[i]) for i in range(x.shape[0])])

    def eval_sens(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.is_cuda and not self.xref.is_cuda:
            self.xref = self.xref.cuda()
            self.xweight = self.xweight.cuda()
        xdiff = x - self.xref
        return self.xweight.mm(xdiff.transpose(0, 1)).transpose(1, 0)

    def eval_hess(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.is_cuda and not self.xref.is_cuda:
            self.xref = self.xref.cuda()
            self.xweight = self.xweight.cuda()
        return torch.stack([self.xweight for i in range (x.shape[0])]) 

    def eval_param_sens(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return NotImplementedError()


class PendulumStageCost(maafa.cost.StageCost):
    def __init__(self, gamma, dt):
        super().__init__()
        self.gamma = gamma
        self.dt = dt
        self.xuref_true = torch.Tensor([1., 0., 0.])
        self.xuweight_true = torch.Tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.001]])
        self.xuref = Variable(self.xuref_true) 
        self.xuweight = Variable(self.xuweight_true)

    def eval(self, x, u, stage):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        assert stage >= 0
        if x.is_cuda and not self.params.is_cuda:
            self.xuref = self.xuref.cuda()
            self.xuweight = self.xuweight.cuda()
        xu = torch.cat([x.transpose(0, 1), u.transpose(0, 1)]).transpose(1, 0)
        xudiff = xu - self.xuref
        Wxudiff = self.xuweight.mm(xudiff.transpose(0, 1)).transpose(1, 0)
        discount = self.gamma **stage
        return discount * self.dt * 0.5 * torch.stack([xudiff[i].dot(Wxudiff[i]) for i in range(x.shape[0])])

    def eval_sens(self, x, u, stage):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        if x.is_cuda and not self.params.is_cuda:
            self.xuref = self.xuref.cuda()
            self.xuweight = self.xuweight.cuda()
        xu = torch.cat([x.transpose(0, 1), u.transpose(0, 1)]).transpose(1, 0)
        xudiff = xu - self.xuref
        discount = self.gamma **stage
        return discount * self.dt * self.xuweight.mm(xudiff.transpose(0, 1)).transpose(1, 0)

    def eval_hess(self, x, u, stage):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        if x.is_cuda and not self.params.is_cuda:
            self.xuref = self.xuref.cuda()
            self.xuweight = self.xuweight.cuda()
        discount = self.gamma **stage
        return discount * self.dt * torch.stack([self.xuweight for i in range (x.shape[0])]) 

    def eval_param_sens(self, x, u, stage):
        return NotImplementedError()