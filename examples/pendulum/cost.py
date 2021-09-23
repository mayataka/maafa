import torch
from torch.autograd import Function, Variable 
from maafa.cost import TerminalCost
from torch import nn

from maafa.cost import TerminalCost, StageCost
from maafa import utils


class PendulumTerminalCost(TerminalCost):
    def __init__(self):
        super().__init__()
        self.xref_true = torch.Tensor([1., 0., 0.])
        self.xweight_true = torch.Tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.xref = Variable(self.x_ref_true) 
        self.xweight = Variable(self.x_weight_true)

    def eval(self, x):
        assert x.dim() == 2 or x.dim() == 1
        if x.dim() == 2:
            xdiff = torch.Tensor([x[j] - self.xref for j in range(x.shape[0])])
            return 0.5 * utils.bmv(xdiff, utils.bmv())
        elif x.dim() == 1:
            xdiff = x - self.xref
            return 0.5 * xdiff.mv(self.xweight.mv(xdiff))
        else:
            return NotImplementedError()

    def eval_sens(self, x):
        assert x.dim() == 2 or x.dim() == 1
        if x.dim() == 2:
            xdiff = torch.Tensor([x[j] - self.xref for j in range(x.shape[0])])
            return utils.bmv()
        elif x.dim() == 1:
            xdiff = x - self.xref
            return self.xweight.mv(xdiff)
        else:
            return NotImplementedError()

    def eval_hess(self, x):
        assert x.dim() == 2 or x.dim() == 1
        if x.dim() == 2:
            return utils.bmv()
        elif x.dim() == 1:
            return self.xweight
        else:
            return NotImplementedError()

    def eval_param_sens(self, x):
        return NotImplementedError()


class PendulumStageCost(StageCost):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        self.xuref_true = torch.Tensor([1., 0., 0., 0.])
        self.xuweight_true = torch.Tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 0.001]])
        self.xuref = Variable(self.xuref_true) 
        self.xuweight = Variable(self.xuweight_true)

    def eval(self, x, u):
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

        xdiff = torch.Tensor([x[j] - self.xref for j in range(x.shape[0])])
        return 0.5 * utils.bmv(xdiff, utils.bmv())


    def eval_sens(self, x, u, lx, lu):
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

        xdiff = torch.Tensor([x[j] - self.xref for j in range(x.shape[0])])
        return 0.5 * utils.bmv(xdiff, utils.bmv())


    def eval_hess(self, x, u):
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

        return self.xuweight

    # def eval_param_sens(self, x, u, dp):
    #     return NotImplementedError()