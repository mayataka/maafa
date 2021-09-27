import torch
from torch.nn.parameter import Parameter

import numpy as np
import math

import os
import tempfile

import maafa
from pendulum.dynamics import PendulumDynamics
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
    params_mpc = PendulumParams()
    params_mpc.dyn_params = Parameter(torch.Tensor([10.0, 0.3, 0.3]))
    params_mpc.xfref = params_mpc.xfref.data # do not treat xfref as a learning parameter
    params_mpc.xuref = params_mpc.xuref.data # do not treat xuref as a learning parameter
    dynamics = PendulumDynamics(dt, params_mpc)
    terminal_cost = maafa.cost.QuadraticTerminalCost(params_mpc)
    stage_cost = maafa.cost.QuadraticStageCost(dt, discount_factor, params_mpc)
    mpc = maafa.MPC(dynamics, stage_cost, terminal_cost, N, nbatch=1, device=device)

    # simulation model
    true_params = PendulumParams()
    model = PendulumDynamics(dt, true_params)

    print("MPC parameters before Q-learning:")
    print(list(mpc.parameters()))

    # ### Off-policy Q-learning 
    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(mpc.parameters(), lr=1.0e-03)
    # maafa.q_learning.train_off_policy(env=model, mpc=mpc, 
    #                                   mpc_sim_steps=1, 
    #                                   mpc_sim_batch_size=1024, 
    #                                   mpc_iter_max=10, 
    #                                   train_mini_batch_size=16, 
    #                                   train_iter_per_episode=10, 
    #                                   loss_fn=loss_fn, 
    #                                   optimizer=optimizer, 
    #                                   episodes=10, verbose_level=2)

    ### On-policy (on-line) Q-learning (the MPC parameters are updated after the each MPC step)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mpc.parameters(), lr=1.0e-03)
    maafa.q_learning.train_on_policy(env=model, mpc=mpc, 
                                     mpc_sim_steps=math.floor(5.0/dt), 
                                     mpc_sim_batch_size=4,
                                     mpc_iter_max=10, 
                                     loss_fn=loss_fn, 
                                     optimizer=optimizer, 
                                     episodes=200, verbose_level=1)

    print("MPC parameters after Q-learning:")
    print(list(mpc.parameters()))

    # number of the batch MPC simulations
    nbatch = 16
    mpc.set_nbatch(nbatch)

    # initial states
    x0 = model.reset(nbatch, device=device)

    # Simulate MPC with learned parameters 
    sim_time = 5.
    sim_step = math.floor(sim_time / dt)
    mpc_iter_max = 10
    x = x0
    tmp_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(tmp_dir))

    for t in range(sim_step):
        u = mpc.mpc_step(x, iter_max=mpc_iter_max)
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