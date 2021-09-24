import torch
from torch.nn.parameter import Parameter

import numpy as np
import math

import os
import tempfile

import maafa
from pendulum.dynamics import PendulumDynamics
from pendulum.cost import PendulumTerminalCost, PendulumStageCost
from pendulum.params import PendulumParams

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # number of the batch MPC simulations
    nbatch = 16

    # setup MPC 
    T = 0.5
    N = 10
    dt = T / N
    discount_factor = 0.99 
    dynamics = PendulumDynamics(dt)
    terminal_cost = PendulumTerminalCost()
    stage_cost = PendulumStageCost(dt, discount_factor)
    mpc = maafa.MPC(dynamics, stage_cost, terminal_cost, N, nbatch=nbatch, device=device)

    # initial states
    torch.manual_seed(0)
    x0 = np.pi*torch.rand(nbatch, dynamics.dimx, device=device)

    # simulation model
    params_true = PendulumParams()
    params_true.dyn_params = torch.Tensor((1.0, 2.5, 2.0))
    model = PendulumDynamics(dt, params=params_true)

    # Dynamics and cost params
    params = PendulumParams(dyn_params=Parameter(dynamics.default_params.to(device)),
                            xuref=Parameter(stage_cost.default_xuref.to(device)),
                            xuweight=Parameter(stage_cost.default_xuweight.to(device)),
                            xfref=Parameter(terminal_cost.default_xfref.to(device)),
                            xfweight=Parameter(terminal_cost.default_xfweight.to(device)))
    mpc.set_params(params)
    print("MPC parameters before Q-learning:")
    print(list(mpc.parameters()))

    loss_fn = torch.nn.SmoothL1Loss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(mpc.parameters(), lr=learning_rate)
    QL_mpc_sim_steps = math.floor(5.0/dynamics.dt) 
    QL_batch_size = 16
    QL_mac_iter_max = 10
    maafa.q_learning.train(model, mpc, QL_mpc_sim_steps, QL_batch_size, 
                           QL_mac_iter_max, loss_fn=loss_fn, 
                           optimizer=optimizer, episodes=10, verbose=True)

    print("MPC parameters after Q-learning:")
    print(list(mpc.parameters()))

    # MPC simulation 
    sim_time = 5.
    sim_step = math.floor(sim_time / dt)
    MPC_iter_max = 10
    x = x0
    tmp_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(tmp_dir))

    for t in range(sim_step):
        u, V = mpc.mpc_step(x, params=params, iter_max=MPC_iter_max)
        x = model.eval(x, u)
        # save figs
        nrow, ncol = 4, 4
        fig, axs = plt.subplots(nrow, ncol, figsize=(3*ncol,3*nrow))
        axs = axs.reshape(-1)
        for i in range(nbatch):
            model.get_frame(x[i], ax=axs[i])
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_dir, '{:03d}.png'.format(t)))
        plt.close(fig)

    # save video
    vid_fname = 'pendulum_q_learning.mp4'
    if os.path.exists(vid_fname):
        os.remove(vid_fname)
    cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        tmp_dir, vid_fname
    )
    os.system(cmd)
    print('Saving video to: {}'.format(vid_fname))