import torch
import copy

import unrestricted_predictor as up
from restricted_predictors import retrain as r
from restricted_predictors import constant_series as cs
from restricted_predictors import zero_attention_score as zas


class Causalities:

    def __init__(self, time_series, model_parameters):
        self.time_series = time_series
        self.train_series = None
        self.test_series = None
        self.model_parameters = model_parameters
        self.base_predictor = None
        self.upper_bound_loss = None
        self.train_base_model()

    def train_base_model(self):
        print("training the unrestricted predictor")

        num_samples, num_series, time_steps = self.time_series.shape
        train_split = int(0.8 * num_samples)
        self.train_series = self.time_series[:train_split]
        self.test_series = self.time_series[train_split:]

        self.base_predictor = up.UnrestrictedPredictor(self.model_parameters, self.train_series, self.test_series)
        self.base_predictor.train_network()
        self.upper_bound_loss = self.base_predictor.predict()

    @staticmethod
    def calculate_directed_information(restricted_loss, upper_bound_loss):
        directed_information = []
        for i in range(len(restricted_loss)):
            if restricted_loss[i] > 0 and upper_bound_loss[i] > 0:
                directed_information.append(restricted_loss[i] / upper_bound_loss[i])
            elif restricted_loss[i] < 0 and upper_bound_loss[i] < 0:
                directed_information.append(upper_bound_loss[i] / restricted_loss[i])
            else:
                directed_information.append(torch.inf)
        directed_information = torch.tensor(directed_information)
        return torch.log(directed_information)

    def generate_causalities(self, method, threshold):
        if method == "retrain":
            retrain = r.Retrain(self.model_parameters, self.train_series, self.test_series)
            restricted_losses = retrain.restricted_networks(self.base_predictor.model)

        elif method == "constant series":
            constant_series = cs.ConstantSeries(self.model_parameters, self.train_series, self.test_series)
            restricted_losses = constant_series.restricted_predictor(self.base_predictor.model)
        elif method == "zero attention score":
            zero_attention_score = zas.ZeroAttentionScore(self.model_parameters, self.train_series, self.test_series)
            restricted_losses = zero_attention_score.restricted_predictor(self.base_predictor.model)
        else:
            print("not a valid method")
            return None

        causalities = []
        for i in range(len(restricted_losses)):
            restricted_loss = restricted_losses[i]
            if method == "retrain":
                restricted_loss = torch.cat(
                    (restricted_loss[:i], torch.tensor([torch.inf]).to(self.base_predictor.device), restricted_loss[i:]))
            directed_information = self.calculate_directed_information(restricted_loss, self.upper_bound_loss)

            print("restricted loss:", i, restricted_loss)
            print("directed information:", i, directed_information)
            print("////////////////////////////////////////////")

            causality = (directed_information > threshold)
            causalities.append(causality)

        print("upper bound loss:", self.upper_bound_loss)
        return causalities

