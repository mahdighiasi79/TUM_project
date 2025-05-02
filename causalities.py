import torch

import unrestricted_predictor as up
from restricted_predictors import retrain as r
from restricted_predictors import constant_series as cs
from restricted_predictors import zero_attention_score as zas


threshold = 0.5


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


def generate_causalities(time_series, method, model_parameters):
    num_samples, num_series, time_steps = time_series.shape
    train_split = int(0.8 * num_samples)
    train_series = time_series[:train_split]
    test_series = time_series[train_split:]

    base_predictor = up.UnrestrictedPredictor(model_parameters, train_series, test_series)

    if method == "retrain":
        retrain = r.Retrain(model_parameters, train_series, test_series)
        upper_bound_loss, restricted_losses = retrain.restricted_networks()

    elif method == "constant series":
        constant_series = cs.ConstantSeries(model_parameters, train_series, test_series)
        upper_bound_loss, restricted_losses = constant_series.restricted_predictor()
    elif method == "zero attention score":
        zero_attention_score = zas.ZeroAttentionScore(model_parameters, train_series, test_series)
        upper_bound_loss, restricted_losses = zero_attention_score.restricted_predictor()
    else:
        print("not a valid method")
        return None

    causalities = []
    for i in range(len(restricted_losses)):
        restricted_loss = restricted_losses[i]
        if method == "retrain":
            restricted_loss = torch.cat(
                (restricted_loss[:i], torch.tensor([torch.inf]).to(base_predictor.device), restricted_loss[i:]))
        directed_information = calculate_directed_information(restricted_loss, upper_bound_loss)

        print("restricted loss:", i, restricted_loss)
        print("directed information:", i, directed_information)
        print("////////////////////////////////////////////")

        causality = (directed_information > threshold)
        causalities.append(causality)

    print("upper bound loss:", upper_bound_loss)
    return causalities

