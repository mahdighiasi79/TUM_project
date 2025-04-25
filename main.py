import torch

from network import tactis


if __name__ == "__main__":
    encoder_dict = {
        "flow_temporal_encoder": {
            "attention_layers": 3,
            "attention_heads": 3,
            "attention_dim": 6,
            "attention_feedforward_dim": 2,
            "dropout": 0.0,
        },
    }

    model_parameters = {
        "flow_series_embedding_dim": 2,
        "flow_input_encoder_layers": 2,
        "bagging_size": None,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
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

    batch_size = 7
    time_steps = 6
    x = torch.randn(batch_size, 1, time_steps)
    y = torch.randn(batch_size, 1, time_steps)
    z = x + y
    z0 = torch.randn(batch_size, 1, 1)
    z = torch.cat([z0, z], dim=2)
    z = z[:, :, :-1]
    input_tokens = torch.cat([x, y, z], dim=1)

    hist_time = torch.arange(time_steps - 1).repeat(batch_size, 3, 1)
    hist_value = input_tokens[:, :, :-1]
    pred_time = torch.tensor([time_steps - 1]).repeat(batch_size, 3, 1)
    pred_value = input_tokens[:, :, -1].reshape(batch_size, 3, 1)

    model = tactis.TACTiS(3, **model_parameters)

    print(model.loss(hist_time, hist_value, pred_time, pred_value))
