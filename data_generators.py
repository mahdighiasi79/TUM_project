import torch
import math
import numpy as np


class HandCraftedSystems:

    def __init__(self, system):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.x1 = torch.randn(3, ).to(self.device)
        self.x2 = torch.randn(3, ).to(self.device)
        self.x3 = torch.randn(3, ).to(self.device)
        self.x4 = torch.randn(3, ).to(self.device)
        self.system = system
        self.generated_data = None

    def next1(self, time_step):
        x1 = torch.sin((0.1 * time_step) + (0.5 * self.x2[time_step - 1])) + torch.log(
            1 + torch.abs(self.x4[time_step - 2])) + torch.randn(1, ).item()
        x2 = torch.tanh(self.x1[time_step - 1]) + (torch.exp(torch.pow(-self.x4[time_step - 2], 2)) / (
                1 + torch.abs(self.x4[time_step - 3]))) + torch.randn(1, ).item()
        x3 = (torch.pow(self.x1[time_step - 2], 2) / (
                1 + torch.exp(-self.x1[time_step - 3]))) + torch.randn(1, ).item()
        x4 = torch.cos(torch.pow(self.x3[time_step - 1], 2)) + torch.randn(1, ).item()
        return torch.tensor([x1, x2, x3, x4]).to(self.device)

    def next2(self, time_step):
        x1 = math.sin(0.05 * time_step) + torch.randn(1, ).item()
        x2 = torch.tanh(self.x1[time_step - 1]) + (torch.log(1 + torch.pow(self.x4[time_step - 2], 2)) / (
                1 + torch.abs(self.x4[time_step - 3]))) + torch.randn(1, ).item()
        x3 = (torch.pow(self.x1[time_step - 2], 2) / (1 + torch.exp(-self.x2[time_step - 1]))) + (
                0.3 * torch.exp(-torch.pow(self.x4[time_step - 1], 2))) + torch.randn(1, ).item()
        x4 = torch.log(1 + torch.pow(self.x3[time_step - 1], 2)) + torch.randn(1, ).item()
        return torch.tensor([x1, x2, x3, x4]).to(self.device)

    def next3(self, time_step):
        x1 = math.sin(0.05 * time_step) + (0.4 * torch.cos(torch.pow(self.x4[time_step - 1], 2))) + torch.randn(
            1, ).item()
        x2 = (torch.log(1 + torch.pow(self.x1[time_step - 1], 2)) / (1 + torch.exp(-self.x3[time_step - 2]))) + (
                0.3 * torch.pow(self.x4[time_step - 3], 3)) + torch.randn(1, ).item()
        x3 = torch.tanh(self.x2[time_step - 1] * self.x1[time_step - 2]) + torch.pow(
            torch.abs(self.x4[time_step - 2]) + 1,
            0.5) + torch.randn(1, ).item()
        x4 = (torch.sin(torch.pow(self.x3[time_step - 1], 2)) / (
                1 + torch.pow(torch.abs(self.x2[time_step - 3]), 1.2))) + (
                     0.2 * torch.exp(-torch.pow(self.x1[time_step - 2], 2))) + torch.randn(1, ).item()
        return torch.tensor([x1, x2, x3, x4]).to(self.device)

    def generate_data(self, n):
        self.x1 = torch.randn(3, ).to(self.device)
        self.x2 = torch.randn(3, ).to(self.device)
        self.x3 = torch.randn(3, ).to(self.device)
        self.x4 = torch.randn(3, ).to(self.device)

        generation_function = None
        if self.system == 1:
            generation_function = self.next1
        elif self.system == 2:
            generation_function = self.next2
        elif self.system == 3:
            generation_function = self.next3

        for i in range(n):
            next_time_step = generation_function(i + 3)
            self.x1 = torch.cat((self.x1, next_time_step[0].unsqueeze(0)))
            self.x2 = torch.cat((self.x2, next_time_step[1].unsqueeze(0)))
            self.x3 = torch.cat((self.x3, next_time_step[2].unsqueeze(0)))
            self.x4 = torch.cat((self.x4, next_time_step[3].unsqueeze(0)))
        self.x1 = self.x1[3:]
        self.x2 = self.x2[3:]
        self.x3 = self.x3[3:]
        self.x4 = self.x4[3:]
        self.generated_data = torch.stack([self.x1, self.x2, self.x3, self.x4]).to(self.device)
        return self.generated_data


class Cut_V:

    def __init__(self, m, v, nonlinear_function):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.m = m
        self.v = v
        self.A = None
        self.x_t = torch.zeros((m,)).to(self.device)
        self.generated_data = torch.tensor([]).to(self.device)
        self.causal_graph = torch.zeros(m, m)
        self.generate_A()

        if nonlinear_function == 'relu':
            self.nonlinear_function = torch.relu
        elif nonlinear_function == 'sigmoid':
            self.nonlinear_function = torch.sigmoid
        elif nonlinear_function == 'elu':
            self.nonlinear_function = torch.nn.ELU(0.1)
        elif nonlinear_function == 'cut_v':
            self.nonlinear_function = self.cut_v
        elif nonlinear_function == 's_r':
            self.nonlinear_function = self.s_r
        elif nonlinear_function == 's_e':
            self.nonlinear_function = self.s_e
        else:
            print("invalid nonlinear function")
            self.nonlinear_function = None

    def generate_A(self):
        self.A = torch.randint(-1, 2, (self.m, self.m, self.m)) * 0.5
        # self.A = torch.ones(self.m, self.m, self.m) * 10
        self.A[2, 0, :] = 0
        self.A[2, :, 0] = 0
        self.A[3, 0, :] = 0
        self.A[3, :, 0] = 0
        self.A = self.A.to(self.device)

    def true_causal_graph(self):
        for i in range(self.m):
            for j in range(self.m):
                for k in range(self.m):
                    if self.A[i, j, k] != 0 or self.A[i, k, j] != 0:
                        self.causal_graph[i, j] = 1
        return self.causal_graph

    def cut_v(self, matrix):
        outer_bound = ((matrix > self.v) + (matrix < -self.v)) * self.v
        inner_bound = ((matrix < self.v) * (matrix > -self.v)) * matrix
        return outer_bound + inner_bound

    @staticmethod
    def s_r(matrix):
        return torch.relu((torch.sigmoid(matrix) - 0.5) * 2)

    @staticmethod
    def s_e(matrix):
        elu = torch.nn.ELU(0.1)
        return elu((torch.sigmoid(matrix) - 0.5) * 2)

    def next(self):
        AX = torch.matmul(self.A, self.x_t)
        self.x_t = torch.einsum('i,ij->j', self.x_t, AX)
        self.x_t += torch.randn(self.m, ).to(self.device)
        self.x_t = self.nonlinear_function(self.x_t)

    def generate_data(self, n):
        self.generated_data = torch.tensor([]).to(self.device)
        for _ in range(n):
            self.next()
            self.generated_data = torch.cat([self.generated_data, self.x_t.unsqueeze(0)], dim=0)
        return self.generated_data


def generate_nonlinear_time_varying_data(T=4000, seed=42):
    np.random.seed(seed)

    # Initialize time series
    X = np.zeros(T)
    Y = np.zeros(T)
    Z = np.zeros(T)

    # Simulate system
    for t in range(1, T):
        # Time-varying coefficient for Y → X
        a = np.sin(2 * np.pi * t / 200)  # cyclic pattern every 200 steps

        # Nonlinear, time-varying causal relationship: Y → X
        X[t] = np.tanh(a * Y[t - 1]) + 0.2 * np.random.randn()

        # Nonlinear static causal relationship: Z → Y
        Y[t] = np.sin(Z[t - 1]) + 0.2 * np.random.randn()

        # Z is independent (no parents)
        Z[t] = 0.8 * np.tanh(Z[t - 1]) + 0.1 * np.random.randn()

    x = torch.tensor(X.reshape(1000, 4), dtype=torch.float)
    y = torch.tensor(Y.reshape(1000, 4), dtype=torch.float)
    z = torch.tensor(Z.reshape(1000, 4), dtype=torch.float)

    input_tokens = torch.stack([x, y, z], dim=1)
    return input_tokens


def generate_static_nonlinear_data(T=4000, seed=42):
    np.random.seed(seed)

    X = np.zeros(T)
    Y = np.zeros(T)
    Z = np.zeros(T)

    for t in range(1, T):
        # Exogenous driver
        Z[t] = 1 / (1 + np.exp(-Z[t - 1])) + 0.3 * np.random.randn()

        # Nonlinear causality: Z → Y
        Y[t] = np.cos(Z[t - 1]) + 0.3 * np.random.randn()

        # Nonlinear causality: Y → X
        X[t] = np.tanh(Y[t - 1]) + 0.3 * np.random.randn()

    x = torch.tensor(X.reshape(2000, 2), dtype=torch.float)
    y = torch.tensor(Y.reshape(2000, 2), dtype=torch.float)
    z = torch.tensor(Z.reshape(2000, 2), dtype=torch.float)

    input_tokens = torch.stack([x, y, z], dim=1)
    return input_tokens
