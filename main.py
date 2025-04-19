import torch

from TACTiS2.tactis import TACTiS


if __name__ == "__main__":
    encoder_dict = {
        "flow_temporal_encoder": {
            "attention_layers": 3,
            "attention_heads": 3,
            "attention_dim": 6,
            "attention_feedforward_dim": 2,
            "dropout": 0.0,
        },
        "copula_temporal_encoder": {
            "attention_layers": 3,
            "attention_heads": 3,
            "attention_dim": 6,
            "attention_feedforward_dim": 2,
            "dropout": 0.0,
        },
    }

    model_parameters = {
        "flow_series_embedding_dim": 2,
        "copula_series_embedding_dim": 2,
        "flow_input_encoder_layers": 2,
        "copula_input_encoder_layers": 2,
        "bagging_size": None,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
        "loss_normalization": "both",
        "positional_encoding": {
            "dropout": 0.0,
        },
        **encoder_dict,
        "copula_decoder": {
            # flow_input_dim and copula_input_dim are passed by the TACTIS module dynamically
            "min_u": 0.0,
            "max_u": 1.0,
            "attentional_copula": {
                "attention_heads": 3,
                "attention_layers": 3,
                "attention_dim": 6,
                "mlp_layers": 1,
                "mlp_dim": 5,
                "resolution": 6,
                "attention_mlp_class": "_easy_mlp",
                "dropout": 0.0,
                "activation_function": "relu",
            },
            "dsf_marginal": {
                "mlp_layers": 2,
                "mlp_dim": 6,
                "flow_layers": 2,
                "flow_hid_dim": 8,
            },
        },
        "experiment_mode": "forecasting",
        "skip_copula": True,
    }

    model = TACTiS(3, **model_parameters)

    hist_time = torch.tensor([[[0, 1], [0, 1], [0, 1]]])
    hist_value = torch.tensor([[[1.0, 2.0], [-1.0, 3.0], [3.0, 4.0]]])
    pred_time = torch.tensor([[[2]]])
    pred_value = torch.tensor([[[5.0]]])

    print(model.loss(hist_time, hist_value, pred_time, pred_value))
