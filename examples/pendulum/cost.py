import torch
from torch.autograd import Variable 


class PendulumTerminalCost(torch.nn.Module):
    def __init__(self, params=None):
        super(PendulumTerminalCost, self).__init__()
        self.default_xfref = torch.Tensor([0., 0.])
        self.default_xfweight = torch.Tensor([[1., 0.], [0., 0.1,]])
        if params is not None:
            if params.xfref is not None:
                self.xfref = params.xfref
            else:
                self.xfref = Variable(self.default_xfref) 
            if params.xfweight is not None:
                self.xfweight = params.xfweight
            else:
                self.xfweight = Variable(self.default_xfweight) 
        else: 
            self.xfref = Variable(self.default_xfref) 
            self.xfweight = Variable(self.default_xfweight) 

    def set_params(self, params):
        if params is not None and params.xfref is not None:
            self.xfref = params.xfref
        else:
            if hasattr(self.xfref, "requires_grad"):
                self.xfref.requires_grad = False 
        if params is not None and params.xfweight is not None:
            self.xfweight = params.xfweight
        else:
            if hasattr(self.xfweight, "requires_grad"):
                self.xfweight.requires_grad = False 

    def eval(self, x, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.set_params(params)
        if x.is_cuda and not self.xfref.is_cuda:
            self.xfref = self.xfref.cuda()
            self.xfweight = self.xfweight.cuda()
        xdiff = x - self.xfref
        Wxdiff = self.xfweight.mm(xdiff.transpose(0, 1)).transpose(1, 0)
        return 0.5 * torch.stack([xdiff[i].dot(Wxdiff[i]) for i in range(x.shape[0])])

    def eval_sens(self, x, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.set_params(params)
        if x.is_cuda and not self.xfref.is_cuda:
            self.xfref = self.xfref.cuda()
            self.xfweight = self.xfweight.cuda()
        xdiff = x - self.xfref
        return self.xfweight.mm(xdiff.transpose(0, 1)).transpose(1, 0)

    def eval_hess(self, x, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.set_params(params)
        if x.is_cuda and not self.xfref.is_cuda:
            self.xfref = self.xfref.cuda()
            self.xfweight = self.xfweight.cuda()
        return torch.stack([self.xfweight for i in range (x.shape[0])]) 

    def forward(self, x, params):
        return self.eval(x, params)


class PendulumStageCost(torch.nn.Module):
    def __init__(self, dt, gamma, params=None):
        super(PendulumStageCost, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.default_xuref = torch.Tensor([0., 0., 0.])
        self.default_xuweight = torch.Tensor([[1., 0., 0.], [0., 0.1, 0.], [0., 0., 0.001]])
        if params is not None:
            if params.xuref is not None:
                self.xuref = params.xuref
            else:
                self.xuref = Variable(self.default_xuref) 
            if params.xuweight is not None:
                self.xuweight = params.xuweight
            else:
                self.xuweight = Variable(self.default_xuweight) 
        else: 
            self.xuref = Variable(self.default_xuref) 
            self.xuweight = Variable(self.default_xuweight) 

    def set_params(self, params):
        if params is not None and params.xuref is not None:
            self.xuref = params.xuref
        else:
            if hasattr(self.xuref, "requires_grad"):
                self.xuref.requires_grad = False 
        if params is not None and params.xuweight is not None:
            self.xuweight = params.xuweight
        else:
            if hasattr(self.xuweight, "requires_grad"):
                self.xuweight.requires_grad = False 

    def eval(self, x, u, stage, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        assert stage >= 0
        self.set_params(params)
        if x.is_cuda and not self.xuref.is_cuda:
            self.xuref = self.xuref.cuda()
            self.xuweight = self.xuweight.cuda()
        xu = torch.cat([x.transpose(0, 1), u.transpose(0, 1)]).transpose(1, 0)
        xudiff = xu - self.xuref
        Wxudiff = self.xuweight.mm(xudiff.transpose(0, 1)).transpose(1, 0)
        discount = self.gamma**stage
        return discount * self.dt * 0.5 * torch.stack([xudiff[i].dot(Wxudiff[i]) for i in range(x.shape[0])])

    def eval_sens(self, x, u, stage, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        self.set_params(params)
        if x.is_cuda and not self.xuref.is_cuda:
            self.xuref = self.xuref.cuda()
            self.xuweight = self.xuweight.cuda()
        xu = torch.cat([x.transpose(0, 1), u.transpose(0, 1)]).transpose(1, 0)
        xudiff = xu - self.xuref
        discount = self.gamma**stage
        return discount * self.dt * self.xuweight.mm(xudiff.transpose(0, 1)).transpose(1, 0)

    def eval_hess(self, x, u, stage, params=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        assert x.dim() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 2
        assert u.shape[1] == 1
        assert u.dim() == 2
        self.set_params(params)
        if x.is_cuda and not self.xuref.is_cuda:
            self.xuref = self.xuref.cuda()
            self.xuweight = self.xuweight.cuda()
        discount = self.gamma**stage
        return discount * self.dt * torch.stack([self.xuweight for i in range (x.shape[0])]) 

    def forward(self, x, u, stage, params):
        return self.eval(x, u, stage, params)