import unrestricted_predictor as up


class Retrain(up.UnrestrictedPredictor):

    def __init__(self, model_parameters, input_tokens, test_tokens):
        super().__init__(model_parameters)
        self.input_tokens = input_tokens
        self.test_tokens = test_tokens
        self.restricted_models = []
        self.restricted_losses = []

    def restricted_networks(self):
        num_samples, num_series, time_steps = self.input_tokens.shape
        for i in range(num_series):
            indices = [j for j in range(self.input_tokens.size(1)) if j != i]
            restricted_tokens = self.input_tokens[:, indices, :]
            restricted_test_tokens = self.test_tokens[:, indices, :]
            self.restricted_models.append(self.train_network(restricted_tokens))
            self.restricted_losses.append(self.predict(restricted_test_tokens))
        return self.restricted_losses
