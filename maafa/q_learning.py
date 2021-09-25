import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np


def mpc_episode(env, mpc, mpc_sim_steps, mpc_sim_batch_size, mpc_iter_max):
    x = env.reset(mpc_sim_batch_size, mpc.device)
    mpc.set_nbatch(mpc_sim_batch_size)
    xm = []
    um = []
    Lm = []
    xm1 = []
    for t in range(mpc_sim_steps):
        u, V = mpc.mpc_step(x, params=None, iter_max=mpc_iter_max)
        L = mpc.ocp.stage_cost.eval_true(x, u)
        x1 = env.eval(x, u)
        xm.append(x)
        um.append(u)
        Lm.append(L.detach())
        xm1.append(x1)
        x = x1
    return xm, um, Lm, xm1


class MPCSimDataset(Dataset):
    def __init__(self, xm, um, Lm, xm1):
        super().__init__()
        self.xm = torch.cat(xm)
        self.um = torch.cat(um)
        self.Lm = torch.cat(Lm)
        self.xm1 = torch.cat(xm1)

    def __getitem__(self, index):
        return self.xm[index], self.um[index], self.Lm[index], self.xm1[index]

    def __len__(self):
        return len(self.xm)


def train(env, mpc, mpc_sim_steps, mpc_sim_batch_size=1, mpc_iter_max=10, 
          train_mini_batch_size=1, train_iter_per_episode=1, 
          loss_fn=None, optimizer=None, episodes=100, verbose=False, debug=False):
    if debug:
        torch.autograd.set_detect_anomaly(True)
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(mpc.parameters(), lr=1.0e-03)
    for episode in range(episodes):
        if verbose:
            print("----------- Episode:", episode+1, "-----------")
        xm, um, Lm, xm1 = mpc_episode(env, mpc, mpc_sim_steps, 
                                      mpc_sim_batch_size, mpc_iter_max)
        mpc_data_set = MPCSimDataset(xm, um, Lm, xm1)
        mpc_data_loader = DataLoader(mpc_data_set, 
                                     batch_size=train_mini_batch_size, 
                                     shuffle=True, drop_last=True)
        mpc.set_nbatch(train_mini_batch_size)
        for iter in range(train_iter_per_episode):
            if verbose:
                TD_errors = []
                losses = []
            for x, u, L, x1 in mpc_data_loader:
                mpc.Q_step(x, u)
                Q = mpc.forward(x, u)
                uopt, V = mpc.mpc_step(x1)
                pred = L.detach() + mpc.ocp.stage_cost.gamma * V.detach()
                loss = loss_fn(Q, pred)
                if verbose:
                    TD_errors.append((Q-pred).abs().mean().item())
                    losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            if verbose:
                print("iter:", iter+1, 
                      ", TD error(avg):", sum(TD_errors), 
                      ", loss:", sum(losses))
        if verbose:
            print("MPC parameters after episode", episode+1, ":", list(mpc.parameters()))