import random
import time

import torch

import data_generators as dg


if __name__ == '__main__':
    data_generator = dg.Cut_V(4, 5, "cut_v")
    print(data_generator.generate_data(10).shape)
