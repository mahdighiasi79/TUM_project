import copy

import torch
from network import tactis
import utils


class UnrestrictedPredictor:

    def __init__(self, model_parameters, train_series, test_series):
        self.model_parameters = model_parameters
        self.train_series = train_series
        self.test_series = test_series
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = None
        self.batch_size = 20

    def train_network(self):
        num_samples, num_series, time_steps = self.train_series.shape
        hist_time, hist_value, pred_time, pred_value = utils.resolve_input(self.train_series, self.device)
        num_batches = len(self.train_series) // self.batch_size

        torch.manual_seed(42)
        model = tactis.TACTiS(num_series, **self.model_parameters).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 10
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
            print(self.predict())
        print("/////////////////////////////")
        return model

    def predict(self):
        num_samples, num_series, time_steps = self.test_series.shape
        hist_time, hist_value, pred_time, pred_value = utils.resolve_input(self.test_series, self.device)

        self.model.eval()
        with torch.inference_mode():
            loss_values = torch.zeros(num_series).to(self.device)
            num_batches = len(self.test_series) // self.batch_size
            for i in range(num_batches):
                mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time, mini_batch_pred_value = utils.give_batch(
                    hist_time, hist_value, pred_time, pred_value, self.batch_size, i)
                loss_values += torch.sum(
                    self.model.loss(mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time,
                                    mini_batch_pred_value),
                    dim=0)
        return -loss_values
