import time

import torch

import data_generators as dg


if __name__ == '__main__':
    cut_v = dg.Cut_V(10, 0.5, "sigmoid")
    data = cut_v.generate_data(1000)
    print(data.shape)
