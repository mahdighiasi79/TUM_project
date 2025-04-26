import torch
import pickle
from network import tactis


if __name__ == "__main__":
    encoder_dict = {
        "flow_temporal_encoder": {
            "attention_layers": 3,
            "attention_heads": 3,
            "attention_dim": 6,
            "attention_feedforward_dim": 2,
            "dropout": 0.0,
            "masked_time_series": None,
        },
    }

    model_parameters = {
        "flow_series_embedding_dim": 2,
        "flow_input_encoder_layers": 2,
        "bagging_size": None,
        "input_encoding_normalization": True,
        "data_normalization": "none",
        "loss_normalization": "both",
        "positional_encoding": {
            "dropout": 0.1,
        },
        **encoder_dict,
        "copula_decoder": {
            # flow_input_dim and copula_input_dim are passed by the TACTIS module dynamically
            "min_u": 0.05,
            "max_u": 0.95,
            "dsf_marginal": {
                "mlp_layers": 2,
                "mlp_dim": 6,
                "flow_layers": 2,
                "flow_hid_dim": 8,
            },
        },
    }

    # num_samples = 2000
    # time_steps = 2
    # num_series = 3
    # x = torch.randn(num_samples, 1, time_steps)
    # y = (torch.randn(num_samples, 1, time_steps) * 10) + 10
    # z = x + y
    # z0 = torch.randn(num_samples, 1, 1)
    # z = torch.cat([z0, z], dim=2)
    # z = z[:, :, :-1]
    # input_tokens = torch.cat([x, y, z], dim=1)
    # with open("simple_data.pkl", "wb") as f:
    #     pickle.dump(input_tokens, f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with open("simple_data.pkl", "rb") as f:
        input_tokens = pickle.load(f)
    num_samples, num_series, time_steps = input_tokens.shape
    input_tokens = input_tokens.to(device)

    input_tokens[:, 1, :] = 0

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
            loss = -torch.sum(model.loss(mini_batch_hist_time, mini_batch_hist_value, mini_batch_pred_time, mini_batch_pred_value))
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
                loss_values += torch.sum(model.loss(test_hist_time, test_hist_value, test_pred_time, test_pred_value), dim=0)
            print(-loss_values)
