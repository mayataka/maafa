import torch
from torch.nn.parameter import Parameter


class PendulumParams(object):
    def __init__(self):
        # Dynamics parameters: gravity (g), mass (m), length (l)
        self.dyn_params = Parameter(torch.Tensor([10., 1., 1.]))
        self.dyn_bias = Parameter(torch.Tensor([0., 0.]))
        # Stage cost parameters 
        self.xuref = Parameter(torch.Tensor([0., 0., 0.]))
        self.L_hess = Parameter(torch.Tensor([[1., 0., 0.], 
                                              [0., 0.1, 0.], 
                                              [0., 0., 0.001]]))
        self.L_grad = Parameter(torch.Tensor([0., 0., 0.]))
        self.L_const = Parameter(torch.Tensor([0.]))
        # Terminal cost parameters 
        self.xfref = Parameter(torch.Tensor([0., 0.])) 
        self.Vf_hess =  Parameter(torch.Tensor([[1., 0.], 
                                                [0., 0.1,]]))
        self.Vf_grad = Parameter(torch.Tensor([0., 0.]))
        self.Vf_const = Parameter(torch.Tensor([0.]))