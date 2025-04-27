import time

import torch

import data_generators as dg


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = torch.tensor([7, 8, 9])
    l = []
    l.append(a)
    l.append(b)
    l.append(c)
    print(l)
    l = torch.stack(l)
    print(l)
    l += torch.identity()
