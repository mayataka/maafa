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

def learning()

EPOCH_NUM = 3000
STEP_MAX = 200

EPOCH_NUM = 3000 # エポック数
STEP_MAX = 200 # 最高ステップ数
MEMORY_SIZE = 200 # メモリサイズいくつで学習を開始するか
BATCH_SIZE = 50 # バッチサイズ
EPSILON = 1.0 # ε-greedy法
EPSILON_DECREASE = 0.001 # εの減少値
EPSILON_MIN = 0.1 # εの下限
START_REDUCE_EPSILON = 200 # εを減少させるステップ数
TRAIN_FREQ = 10 # Q関数の学習間隔
UPDATE_TARGET_Q_FREQ = 20 # Q関数の更新間隔
GAMMA = 0.97 # 割引率
LOG_FREQ = 1000 # ログ出力の間隔



if __name__ == '__main__':
    # number of the batch MPC simulations
    nbatch = 16

    # setup MPC 
    T = 0.5
    N = 10
    dt = T / N
    gamma = 1.0 # discount factor
    dynamics = PendulumDynamics(dt)
    terminal_cost = PendulumTerminalCost()
    stage_cost = PendulumStageCost(dt, gamma)
    mpc = maafa.MPC(dynamics, stage_cost, terminal_cost, N, nbatch=nbatch)

    # initial states
    torch.manual_seed(0)
    x0 = np.pi*torch.rand(nbatch, dynamics.dimx)
    xmin = torch.Tensor([-np.pi, -1.])
    xmax = torch.Tensor([np.pi, 1.])
    x0 = torch.clamp(x0, xmin, xmax)

    # simulation model
    model = PendulumDynamics(dt)
    true_params = torch.Tensor((1.0, 2.5, 2.0))
    model.params = true_params

    # MPC simulation 
    sim_time = 5.
    sim_step = math.floor(sim_time / dt)
    MPC_iter_max = 5
    x = x0
    tmp_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(tmp_dir))

    for t in range(sim_step):
        u = mpc.mpc_step(x, iter_max=MPC_iter_max)
        urand = u + torch.rand(nbatch, dynamics.dimu)
        x1 = model.eval(x, u)
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
    vid_fname = 'pendulum_Q_learning.mp4'
    if os.path.exists(vid_fname):
        os.remove(vid_fname)
    cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        tmp_dir, vid_fname
    )
    os.system(cmd)
    print('Saving video to: {}'.format(vid_fname))