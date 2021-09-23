import torch

def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)