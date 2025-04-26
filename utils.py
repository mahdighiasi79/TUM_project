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
