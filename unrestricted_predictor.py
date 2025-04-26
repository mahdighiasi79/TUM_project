import torch
from network import tactis
import utils


class UnrestrictedPredictor:

    def __init__(self, model_parameters, input_tokens):
        self.model_parameters = model_parameters
        self.input_tokens = input_tokens
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = None
        self.batch_size = 20

    def train_network(self):
        num_samples, num_series, time_steps = self.input_tokens.shape
        hist_time, hist_value, pred_time, pred_value = utils.resolve_input(self.input_tokens, self.device)
        num_batches = len(self.input_tokens) // self.batch_size

        torch.manual_seed(42)
        model = tactis.TACTiS(num_series, **self.model_parameters).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 20
        for epoch in range(epochs):
            for i in range(num_batches):
                mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time, mini_batch_pred_value = utils.give_batch(
                    hist_time, hist_value, pred_time, pred_value, self.batch_size, i)

                model.train()
                loss = -torch.sum(model.loss(mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time,
                                             mini_batch_pred_value))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model = model
        return model

    def predict(self, test_series):
        num_samples, num_series, time_steps = test_series.shape
        hist_time, hist_value, pred_time, pred_value = utils.resolve_input(test_series, self.device)

        self.model.eval()
        with torch.inference_mode():
            loss_values = torch.zeros(num_series).to(self.device)
            num_batches = len(test_series) // self.batch_size
            for i in range(num_batches):
                mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time, mini_batch_pred_value = utils.give_batch(
                    hist_time, hist_value, pred_time, pred_value, self.batch_size, i)
                loss_values += torch.sum(
                    self.model.loss(mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time,
                                    mini_batch_pred_value),
                    dim=0)
        return -loss_values
