from network import tactis
import torch


class GroundTruth:

    def __init__(self, model_parameters, input_tokens):
        self.model_parameters = model_parameters
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def train_network(self, input_tokens):
        num_samples, num_series, time_steps = input_series.shape
        hist_time = torch.arange(time_steps - 1).repeat(num_samples, num_series, 1).to(device)
        hist_value = input_tokens[:, :, :-1]
        pred_time = torch.tensor([time_steps - 1]).repeat(num_samples, num_series, 1).to(device)
        pred_value = input_tokens[:, :, -1].reshape(num_samples, num_series, 1)

        batch_size = 20
        train_split = int(0.8 * num_samples)

        train_series = input_tokens[:train_split]
        num_batches = len(train_series) // batch_size
        train_hist_time = hist_time[:train_split]
        train_hist_value = hist_value[:train_split]
        train_pred_time = pred_time[:train_split]
        train_pred_value = pred_value[:train_split]

        test_series = input_tokens[train_split:]
        num_test_batches = len(test_series) // batch_size
        test_hist_time = hist_time[train_split:]
        test_hist_value = hist_value[train_split:]
        test_pred_time = pred_time[train_split:]
        test_pred_value = pred_value[train_split:]

        torch.manual_seed(42)
        model = tactis.TACTiS(num_series, **model_parameters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 20
        for epoch in range(epochs):
            for i in range(num_batches):
                mini_batch = train_series[i * batch_size: (i + 1) * batch_size]
                mini_batch_hist_time = train_hist_time[i * batch_size: (i + 1) * batch_size]
                mini_batch_hist_value = train_hist_value[i * batch_size: (i + 1) * batch_size]
                mini_batch_pred_time = train_pred_time[i * batch_size: (i + 1) * batch_size]
                mini_batch_pred_value = train_pred_value[i * batch_size: (i + 1) * batch_size]

                model.train()
                loss = -torch.sum(model.loss(mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time,
                                             mini_batch_pred_value))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                loss_values = torch.zeros(num_series).to(device)
                for i in range(num_test_batches):
                    mini_batch = test_series[i * batch_size: (i + 1) * batch_size]
                    mini_batch_hist_time = test_hist_time[i * batch_size: (i + 1) * batch_size]
                    mini_batch_hist_value = test_hist_value[i * batch_size: (i + 1) * batch_size]
                    mini_batch_pred_time = test_pred_time[i * batch_size: (i + 1) * batch_size]
                    mini_batch_pred_value = test_pred_value[i * batch_size: (i + 1) * batch_size]
                    loss_values += torch.sum(
                        model.loss(test_hist_time, test_hist_value, test_pred_time, test_pred_value), dim=0)
                print(-loss_values)

    def causal_matrix(time_series, model_parameters):
        time_steps, num_series = time_series.shape
        torch.manual_seed(42)
        model = tactis.TACTiS(num_series, **model_parameters).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
