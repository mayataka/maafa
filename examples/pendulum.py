import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np
import math

import os
import io
import base64
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
    mpc = maafa.MPC(dynamics, stage_cost, terminal_cost, N, nbatch=nbatch)

    print(list(mpc.parameters()))

    # initial states
    torch.manual_seed(0)
    x0 = np.pi*torch.rand(nbatch, dynamics.dimx)

    # simulation model
    model = PendulumDynamics(dt)

    # Dynamics and cost params
    params = PendulumParams(dyn_params=Parameter(torch.Tensor((10., 1., 1.))))

    # MPC simulation 
    sim_time = 5.
    sim_step = math.floor(sim_time / dt)
    MPC_iter_max = 10
    x = x0
    tmp_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(tmp_dir))

    for t in range(sim_step):
        u = mpc.mpc_step(x, params=params, iter_max=MPC_iter_max, verbose=True)
        urand = u + torch.rand(nbatch, dynamics.dimu)
        x1 = model.eval(x, u)
        td = mpc.forward(x, x1, u, params=params)
        x = x1
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
    vid_fname = 'pendulum.mp4'
    if os.path.exists(vid_fname):
        os.remove(vid_fname)
    cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        tmp_dir, vid_fname
    )
    os.system(cmd)
    print('Saving video to: {}'.format(vid_fname))