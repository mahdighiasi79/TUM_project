import copy

import unrestricted_predictor as up


class Retrain(up.UnrestrictedPredictor):

    def __init__(self, model_parameters, train_series, test_series):
        super().__init__(model_parameters, train_series, test_series)
        self.restricted_models = []
        self.restricted_losses = []

    def restricted_networks(self):
        num_samples, num_series, time_steps = self.train_series.shape

        base_model = self.train_network()
        upper_bound_loss = self.predict()

        train_series_copy = copy.deepcopy(self.train_series)
        test_series_copy = copy.deepcopy(self.test_series)
        for i in range(num_series):
            indices = [j for j in range(self.train_series.size(1)) if j != i]
            self.train_series = self.train_series[:, indices, :]
            self.test_series = self.test_series[:, indices, :]

            self.restricted_models.append(self.train_network())
            self.restricted_losses.append(self.predict())

            self.train_series = copy.deepcopy(train_series_copy)
            self.test_series = copy.deepcopy(test_series_copy)

        return upper_bound_loss, self.restricted_losses
