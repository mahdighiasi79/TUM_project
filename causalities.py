import torch

from restricted_predictors import retrain as r
import unrestricted_predictor as up


threshold = 0.5


def generate_causalities(time_series, method, model_parameters):
    num_samples, num_series, time_steps = time_series.shape
    train_split = int(0.8 * num_samples)
    train_series = time_series[:train_split]
    test_series = time_series[train_split:]

    base_predictor = up.UnrestrictedPredictor(model_parameters)
    base_model = base_predictor.train_network(train_series)
    upper_bound_loss = base_predictor.predict(test_series)

    if method == "retrain":
        retrain = r.Retrain(model_parameters, train_series, test_series)
        restricted_losses = retrain.restricted_networks()
        causalities = []
        for i in range(len(restricted_losses)):
            restricted_loss = restricted_losses[i]
            restricted_loss = torch.cat((restricted_loss[:i], torch.tensor([torch.inf]).to(retrain.device), restricted_loss[i:]))
            directed_information = torch.log(restricted_loss / upper_bound_loss)
            causality = (directed_information > threshold)
            causalities.append(causality)
        return causalities
    return None
