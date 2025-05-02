import random
import time
import torch

from network import tactis
import data_generators as dg
import utils


if __name__ == '__main__':
    t = torch.randn(2, 3, 4, 5)
    print(t.transpose(-2, -1).shape)
