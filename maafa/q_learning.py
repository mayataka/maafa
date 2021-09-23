import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np
import math

class QLearning(object):
    def __init__(self, 
                 EPOCH_NUM = 3000,
                 STEP_MAX = 200,
                 MEMORY_SIZE = 200,
                 BATCH_SIZE = 16,
                 EPSILON = 1.0,
                 EPSILON_DECAY = 0.005,
                 EPSILON_MIN = 0.05,
                 START_REDUCE_EPSILON = 200 ,
                 TRAIN_FREQ = 10 ,
                 UPDATE_TARGET_Q_FREQ = 20,
                 GAMMA = 0.99,
                 LOG_FREQ = 1000):

    def train(env, mpc, loss_func=torch.nn.HuberLoss):
        

def Q_learning():
    optimizer = torch.optim.Adam(mpc.parameters(), lr=learning_rate)

    x = x0
    for t in range(steps):
        u = mpc.mpc_step(x, MPC_iter_max)
        x1 = env.eval(x, u)
        TD_error = mpc.forward(x, x1, u, MPC_kkt_tol, MPC_iter_max)
        optimizer.zero_grad()
        TD_error.backward()
        optimizer.step()
