import torch

import numpy as np


def mpc_episode(env, mpc, mpc_sim_steps, batch_size, mpc_iter_max):
    torch.manual_seed(0)
    x = env.reset(batch_size)
    xm = []
    um = []
    Vm = []
    Lm = []
    for t in range(mpc_sim_steps):
        u, V = mpc.mpc_step(x, params=None, iter_max=mpc_iter_max)
        L = mpc.ocp.stage_cost.eval(x, u, stage=0)
        x1 = env.eval(x, u)
        xm.append(x)
        um.append(u)
        Vm.append(V)
        Lm.append(L)
        x = x1
    return xm, um, Vm, Lm

def train(env, mpc, mpc_sim_steps, batch_size, mpc_iter_max, 
          loss_fn=None, optimizer=None, episodes=100, verbose=False, debug=False):
    if debug:
        torch.autograd.set_detect_anomaly(True)
    if loss_fn is None:
        loss_fn = torch.nn.SmoothL1Loss()
    if optimizer is None:
        optimizer = torch.optim.Adam(mpc.parameters(), lr=0.001)
    for episode in range(episodes):
        if verbose:
            print("----------- Episode:", episode+1, "-----------")
        xm, um, Vm, Lm = mpc_episode(env, mpc, mpc_sim_steps, 
                                     batch_size, mpc_iter_max)
        discount_factor = mpc.ocp.stage_cost.gamma
        for t in range(mpc_sim_steps-1):
            mpc.Q_step(xm[t], um[t])
            Qt = mpc.forward(xm[t], um[t])
            pred = Lm[t] + discount_factor * Vm[t+1]
            loss = loss_fn(Qt, pred)
            if verbose:
                print("sim step:", t, ", loss:", loss)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()