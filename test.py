import numpy as np
import torch
from simulated_data import cut_v
from simulated_data import system1
import torch


def relu(matrix):
    result = matrix * (matrix > 0)
    return result


if __name__ == '__main__':
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tensor = tensor.add(5.0)
    print(tensor)

# https://chatgpt.com/share/67fda1e5-4e18-800c-b853-f09555baebed
# https://chatgpt.com/share/67fe239f-f1ec-800c-9c1c-bfb15a70e729

