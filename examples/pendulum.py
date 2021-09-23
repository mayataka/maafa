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
from IPython.display import HTML


import maafa
from pendulum.dynamics import PendulumDynamics
from pendulum.cost import PendulumTerminalCost, PendulumStageCost


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')


if __name__ == '__main__':
    # number of the parallel MPC simulations
    nbatch = 16

    # setup MPC 
    T = 1.0
    N = 20
    dt = T / N
    gamma = 0.99 # discount factor
    dynamics = PendulumDynamics(dt)
    terminal_cost = PendulumTerminalCost()
    stage_cost = PendulumStageCost(dt, gamma)
    mpc = maafa.MPC(dynamics, stage_cost, terminal_cost, N, nbatch=nbatch)

    # initial states
    torch.manual_seed(0)
    x0 = torch.rand(nbatch, dynamics.dimx)
    xmin = torch.Tensor([-(1/2)*np.pi, -1.])
    xmax = torch.Tensor([(1/2)*np.pi, 1.])
    x0 = torch.clamp(x0, xmin, xmax)

    # simulation model
    model = PendulumDynamics(dt)

    # MPC simulation 
    sim_time = 10.
    sim_step = math.floor(sim_time / dt)
    MPC_iter_max = 5
    x = x0
    t_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(t_dir))

    for t in range(sim_step):
        u = mpc.mpc_step(x0, iter_max=MPC_iter_max)
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
        fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
        plt.close(fig)

    # save video
    vid_fname = 'pendulum_swing_up.mp4'
    if os.path.exists(vid_fname):
        os.remove(vid_fname)
    cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        t_dir, vid_fname
    )
    os.system(cmd)
    print('Saving video to: {}'.format(vid_fname))

    video = io.open(vid_fname, 'r+b').read()
    encoded = base64.b64encode(video)
    HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii')))