import copy
import random

import unrestricted_predictor as up


class ZeroAttentionScore(up.UnrestrictedPredictor):

    def __init__(self, model_parameters, train_series, test_series):
        super().__init__(model_parameters, train_series, test_series)
        self.restricted_losses = []

    @staticmethod
    def mask_series(base_model, series_index):
        for attention_layer in base_model.flow_encoder.layer_series:
            attention_layer.self_attn.masked_time_series = series_index

    def restricted_predictor(self, base_model):
        num_samples, num_series, time_steps = self.train_series.shape
        self.model = copy.deepcopy(base_model)
        for i in range(num_series):
            self.model.flow_encoder.layer_series[0].self_attn.masked_time_series = i
            # self.mask_series(base_model, i)
            self.restricted_losses.append(self.predict())
            self.model = copy.deepcopy(base_model)
        return self.restricted_losses
