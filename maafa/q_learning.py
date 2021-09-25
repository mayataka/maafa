import torch
from torch.utils.data import Dataset, DataLoader


def mpc_episode(env, mpc, mpc_sim_steps, mpc_sim_batch_size, mpc_iter_max):
    x = env.reset(mpc_sim_batch_size, mpc.device)
    mpc.set_nbatch(mpc_sim_batch_size)
    xm = []
    um = []
    Lm = []
    xm1 = []
    for t in range(mpc_sim_steps):
        with torch.no_grad():
            u = mpc.mpc_step(x, params=None, iter_max=mpc_iter_max)
            L = mpc.ocp.stage_cost.eval_true(x, u)
            x1 = env.eval(x, u)
            xm.append(x)
            um.append(u)
            Lm.append(L)
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


def train_off_policy(env, mpc, mpc_sim_steps, mpc_sim_batch_size=1, 
                     mpc_iter_max=10, train_mini_batch_size=1, 
                     train_iter_per_episode=1, loss_fn=None, optimizer=None, 
                     learning_rate=1.0e-03, episodes=100, verbose=False, 
                     debug=False):
    torch.autograd.set_detect_anomaly(debug)
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
        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(mpc.parameters(), lr=learning_rate)
        for iter in range(train_iter_per_episode):
            if verbose:
                TD_errors = []
                losses = []
            for x, u, L, x1 in mpc_data_loader:
                with torch.no_grad():
                    mpc.Q_step(x, u)
                Q = mpc.forward(x.detach(), u.detach())
                with torch.no_grad():
                    _u = mpc.mpc_step(x1)
                    V = mpc.forward(x1)
                    Q_expect = L + mpc.ocp.stage_cost.gamma * V
                loss = loss_fn(Q, Q_expect.detach())
                if verbose:
                    TD_errors.append((Q-Q_expect).abs().mean().item())
                    losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            if verbose:
                print("iter:", iter+1, 
                      ", TD error(avg):", sum(TD_errors)/len(TD_errors), 
                      ", loss(avg):", sum(losses)/len(losses))
        if verbose:
            print("MPC parameters after episode", episode+1, ":", list(mpc.parameters()))


def mpc_episode_on_policy(env, mpc, mpc_sim_steps, mpc_sim_batch_size=1, 
                          mpc_iter_max=10, loss_fn=None, optimizer=None, 
                          learning_rate=1.0e-03, verbose=False, debug=False):
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(mpc.parameters(), lr=learning_rate)
    x = env.reset(mpc_sim_batch_size, mpc.device)
    mpc.set_nbatch(mpc_sim_batch_size)
    for t in range(mpc_sim_steps):
        with torch.no_grad():
            u = mpc.mpc_step(x, params=None, iter_max=mpc_iter_max)
        if t > 0:
            with torch.no_grad():
                V = mpc.forward(x.detach())
                Q_expect = L + mpc.ocp.stage_cost.gamma * V
            loss = loss_fn(Q, Q_expect.detach())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if verbose:
                print("t:", t, 
                      ", TD error(avg):", (Q-Q_expect).abs().mean().item(), 
                      ", loss(avg):", loss.item())
        Q = mpc.forward(x.detach(), u.detach())
        with torch.no_grad():
            L = mpc.ocp.stage_cost.eval_true(x, u)
            x1 = env.eval(x, u)
            x = x1


def train_on_policy(env, mpc, mpc_sim_steps, mpc_sim_batch_size=1, 
                    mpc_iter_max=10, loss_fn=None, optimizer=None, 
                    learning_rate=1.0e-03, episodes=100, verbose=False, 
                    debug=False):
    torch.autograd.set_detect_anomaly(debug)
    for episode in range(episodes):
        if verbose:
            print("----------- Episode:", episode+1, "-----------")
        mpc_episode_on_policy(env=env, mpc=mpc, mpc_sim_steps=mpc_sim_steps,
                              mpc_sim_batch_size=mpc_sim_batch_size,
                              mpc_iter_max=mpc_iter_max, loss_fn=loss_fn, 
                              optimizer=optimizer, learning_rate=learning_rate, 
                              verbose=verbose)
        if verbose:
            print("MPC parameters after episode", episode+1, ":", list(mpc.parameters()))