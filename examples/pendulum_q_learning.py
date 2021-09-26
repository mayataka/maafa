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
                            dyn_bias=Parameter(dynamics.default_dyn_bias.to(device)),
                            # xuref=Parameter(stage_cost.default_xuref.to(device)),
                            L_hess=Parameter(stage_cost.default_L_hess.to(device)),
                            L_grad=Parameter(stage_cost.default_L_grad.to(device)),
                            L_const=Parameter(stage_cost.default_L_const.to(device)),
                            # xfref=Parameter(terminal_cost.default_xfref.to(device)),
                            Vf_hess=Parameter(terminal_cost.default_Vf_hess.to(device)),
                            Vf_grad=Parameter(terminal_cost.default_Vf_grad.to(device)),
                            Vf_const=Parameter(terminal_cost.default_Vf_const.to(device)))
    mpc.set_params(params)
    print("MPC parameters before Q-learning:")
    print(list(mpc.parameters()))

    # ### Off-policy Q-learning 
    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.AdamW(mpc.parameters(), lr=1.0e-03, amsgrad=True)
    # maafa.q_learning.train_off_policy(env=model, mpc=mpc, 
    #                                   mpc_sim_steps=math.floor(0.1/dt), 
    #                                   mpc_sim_batch_size=256, 
    #                                   mpc_iter_max=10, 
    #                                   train_mini_batch_size=64, 
    #                                   train_iter_per_episode=20, 
    #                                   loss_fn=loss_fn, 
    #                                   optimizer=optimizer, 
    #                                   episodes=20, verbose=True)

    ### On-policy (on-line) Q-learning (the MPC parameters are updated after each MPC step)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(mpc.parameters(), lr=1.0e-03, amsgrad=True)
    maafa.q_learning.train_on_policy(env=model, mpc=mpc, 
                                     mpc_sim_steps=math.floor(0.1/dt), 
                                     mpc_sim_batch_size=1,
                                     mpc_iter_max=10, 
                                     loss_fn=loss_fn, 
                                     optimizer=optimizer, 
                                     episodes=1000, verbose=True)

    # print("MPC parameters after Q-learning:")
    # print(list(mpc.parameters()))

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