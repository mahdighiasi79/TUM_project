import copy
import random

import unrestricted_predictor as up


class ConstantSeries(up.UnrestrictedPredictor):

    def __init__(self, model_parameters, train_data, test_data):
        super().__init__(model_parameters)
        self.train_data, self.test_data = train_data, test_data
        self.restricted_losses = []
        self.constant = random.randint(-100, 100)

    def restricted_predictor(self):
        num_samples, num_series, time_steps = self.train_data.shape
        self.train_network(self.train_data)
        for i in range(num_series):
            restricted_test_data = copy.deepcopy(self.test_data)
            restricted_test_data[:, i, :] = self.constant
            self.restricted_losses.append(self.predict(restricted_test_data))
        return self.restricted_losses
