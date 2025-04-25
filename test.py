import time

import torch

import data_generators as dg


if __name__ == '__main__':
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t[:, 1] = 0
    print(t)
