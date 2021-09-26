import torch
from torch.nn.parameter import Parameter


def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)

def symmetrize(X):
    return X.triu() + X.triu(1).transpose(-1, -2)

def make_positive_definite(X, eps=1.0e-06):
    return X
    assert eps >= 0.
    Lmd, V = torch.linalg.eig(X)
    Lmd.real = torch.clamp(Lmd.real, min=eps)
    return V.mm(torch.diag(Lmd)).mm(V.transpose(0, 1)).real

def get_data(obj):
    if isinstance(obj, Parameter):
        return obj.data
    else:
        return obj