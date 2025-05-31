import torch


def resolve_input(input_tokens, device):
    num_samples, num_series, time_steps = input_tokens.shape
    hist_time = torch.arange(time_steps - 1).repeat(num_samples, num_series, 1).to(device)
    hist_value = input_tokens[:, :, :-1]
    pred_time = torch.tensor([time_steps - 1]).repeat(num_samples, num_series, 1).to(device)
    pred_value = input_tokens[:, :, -1].reshape(num_samples, num_series, 1)
    return hist_time, hist_value, pred_time, pred_value


def give_batch(hist_time, hist_value, pred_time, pred_value, batch_size, index):
    mini_batch_hist_time = hist_time[index * batch_size: (index + 1) * batch_size]
    mini_batch_hist_value = hist_value[index * batch_size: (index + 1) * batch_size]
    mini_batch_pred_time = pred_time[index * batch_size: (index + 1) * batch_size]
    mini_batch_pred_value = pred_value[index * batch_size: (index + 1) * batch_size]
    return mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time, mini_batch_pred_value


def models_equal(model1, model2):
    if type(model1) is not type(model2):
        return False
    if len(list(model1.parameters())) != len(list(model2.parameters())):
        return False
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            return False
    return True


class Experiment:

    def __init__(self, model_parameters, data, learning_rate, batch_size, epochs):
        self.model_parameters = model_parameters
        self.data = data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
