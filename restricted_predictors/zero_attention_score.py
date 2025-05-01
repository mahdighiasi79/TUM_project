import copy
import random

import unrestricted_predictor as up


class ZeroAttentionScore(up.UnrestrictedPredictor):

    def __init__(self, model_parameters, train_series, test_series):
        super().__init__(model_parameters, train_series, test_series)
        self.restricted_losses = []
        self.constant = random.randint(-100, 100)

    def restricted_predictor(self):
        num_samples, num_series, time_steps = self.train_series.shape
        base_model = self.train_network()
        upper_bound_loss = self.predict()
        for i in range(num_series):
            base_model.flow_encoder.masked_time_series = i
            self.restricted_losses.append(self.predict())
        return upper_bound_loss, self.restricted_losses
