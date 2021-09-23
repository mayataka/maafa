## maafa - MPC as a Function Approximator

This is an un-official PyTorch implementation of a very interesting and groundbreaking paper from S. Grod and M. Zanon:

```
@article{ENMPC-RL,
  author={S. Gros and M. Zanon},
  journal={IEEE Transactions on Automatic Control}, 
  title={Data-Driven Economic NMPC Using Reinforcement Learning}, 
  year={2020},
  volume={65},
  number={2},
  pages={636-648},
  doi={10.1109/TAC.2019.2913768}
}
```

- The MPC problem is discretized by the direct multiple shooting method.
- The KKT system to compute the Newton step is solved by Riccati recursion, which can consider batch implementation.