import copy
import random

import unrestricted_predictor as up


class ConstantSeries(up.UnrestrictedPredictor):

    def __init__(self, model_parameters, train_series, test_series):
        super().__init__(model_parameters, train_series, test_series)
        self.restricted_losses = []
        self.constant = random.randint(-100, 100)

    def restricted_predictor(self):
        num_samples, num_series, time_steps = self.train_series.shape
        self.train_network()
        upper_bound_loss = self.predict()
        test_series_copy = copy.deepcopy(self.test_series)
        for i in range(num_series):
            self.test_series[:, i, :] = self.constant
            self.restricted_losses.append(self.predict())
            self.test_series = copy.deepcopy(test_series_copy)
        return upper_bound_loss, self.restricted_losses
