import copy
import random
import torch

import unrestricted_predictor as up


class ConstantSeries(up.UnrestrictedPredictor):

    def __init__(self, model_parameters, train_series, test_series):
        super().__init__(model_parameters, train_series, test_series)
        self.restricted_losses = []
        self.constant = 0

    def restricted_predictor(self, base_model):
        num_samples, num_series, time_steps = self.train_series.shape
        self.model = base_model
        test_series_copy = copy.deepcopy(self.test_series)
        for i in range(num_series):
            self.constant = (torch.mean(self.test_series[:, i, :]) + 1) * 10
            self.test_series[:, i, :] = self.constant
            self.restricted_losses.append(self.predict())
            self.test_series = copy.deepcopy(test_series_copy)
        return self.restricted_losses
