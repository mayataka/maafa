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

    # setup MPC 
    T = 0.5
    N = 10
    dt = T / N
    discount_factor = 0.99 
    dynamics = PendulumDynamics(dt)
    terminal_cost = PendulumTerminalCost()
    stage_cost = PendulumStageCost(dt, discount_factor)
    mpc = maafa.MPC(dynamics, stage_cost, terminal_cost, N, device=device)

    # simulation model
    model = PendulumDynamics(dt)

    # Dynamics and cost params
    inaccurate_pendulum_params = torch.Tensor((1.0, 0.6, 0.6))
    params = PendulumParams(dyn_params=Parameter(inaccurate_pendulum_params.to(device)),
                            xuref=Parameter(stage_cost.default_xuref.to(device)),
                            xuweight=Parameter(stage_cost.default_xuweight.to(device)),
                            xfref=Parameter(terminal_cost.default_xfref.to(device)),
                            xfweight=Parameter(terminal_cost.default_xfweight.to(device)))
    mpc.set_params(params)
    print("MPC parameters before Q-learning:")
    print(list(mpc.parameters()))

    # ### Off-policy (off-line) Q-learning 
    # loss_fn = torch.nn.MSELoss()
    # learning_rate = 1.0e-03
    # optimizer = torch.optim.Adam(mpc.parameters(), lr=learning_rate)
    # maafa.q_learning.train_off_policy(env=model, mpc=mpc, mpc_sim_steps=1, 
    #                                   mpc_sim_batch_size=16, mpc_iter_max=10, 
    #                                   train_mini_batch_size=1, 
    #                                   train_iter_per_episode=20, 
    #                                   loss_fn=loss_fn, 
    #                                   optimizer=optimizer, 
    #                                   learning_rate=learning_rate,
    #                                   episodes=100, verbose=True)

    ### On-policy (on-line) Q-learning 
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1.0e-03
    optimizer = torch.optim.Adam(mpc.parameters(), lr=learning_rate)
    maafa.q_learning.train_on_policy(env=model, mpc=mpc, 
                                     mpc_sim_steps=math.floor(5.0/dt), 
                                     mpc_sim_batch_size=4, 
                                     mpc_iter_max=10, 
                                     loss_fn=loss_fn, 
                                     optimizer=optimizer, 
                                     learning_rate=learning_rate,
                                     episodes=200, verbose=True)

    print("MPC parameters after Q-learning:")
    print(list(mpc.parameters()))

    # number of the batch MPC simulations
    nbatch = 16
    mpc.set_nbatch(nbatch)

    # initial states
    x0 = np.pi*torch.rand(nbatch, dynamics.dimx, device=device)

    # Simulate MPC with learned parameters 
    sim_time = 5.
    sim_step = math.floor(sim_time / dt)
    MPC_iter_max = 10
    x = x0
    tmp_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(tmp_dir))

    for t in range(sim_step):
        u = mpc.mpc_step(x, params=params, iter_max=MPC_iter_max)
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