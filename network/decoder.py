"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from typing import Any, Dict, Optional, Type
import torch
import copy
from torch import nn
from .marginal import DSFMarginal


def _merge_series_time_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series, time steps, ...] to one with dimensions [batch, series * time steps, ...]
    """
    assert x.dim() >= 3
    return x.view((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])


def _split_series_time_dims(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series * time steps, ...] to one with dimensions [batch, series, time steps, ...]
    """
    assert x.dim() + 1 == len(target_shape)
    return x.view(target_shape)


def _simple_linear_projection(input_dim: int, output_dim: int) -> nn.Sequential:
    layers = [nn.Linear(input_dim, output_dim)]
    return nn.Sequential(*layers)


def _easy_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: Type[nn.Module],
) -> nn.Sequential:
    """
    Generate a MLP with the given parameters.
    """
    elayers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(1, num_layers):
        elayers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    elayers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*elayers)


class CopulaDecoder(nn.Module):
    """
    A decoder which forecast using a distribution built from a copula and marginal distributions.
    """

    def __init__(
        self,
        flow_input_dim: int,
        min_u: float = 0.0,
        max_u: float = 1.0,
        skip_sampling_marginal: bool = False,
        dsf_marginal: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            The dimension of the encoded representation (upstream data encoder).
        min_u: float, default to 0.0
        max_u: float, default to 1.0
            The values sampled from the copula will be scaled from [0, 1] to [min_u, max_u] before being sent to the marginal.
        skip_sampling_marginal: bool, default to False
            If set to True, then the output from the copula will not be transformed using the marginal during sampling.
            Does not impact the other transformations from observed values to the [0, 1] range.
        trivial_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a TrivialCopula.
            The options sent to the TrivialCopula is content of this dictionary.
        attentional_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a AttentionalCopula.
            The options sent to the AttentionalCopula is content of this dictionary.
        dsf_marginal: Dict[str, Any], default to None
            If set to a non-None value, uses a DSFMarginal.
            The options sent to the DSFMarginal is content of this dictionary.
        """
        super().__init__()

        self.flow_input_dim = flow_input_dim
        self.min_u = min_u
        self.max_u = max_u
        self.skip_sampling_marginal = skip_sampling_marginal
        self.dsf_marginal_args = dsf_marginal

        if dsf_marginal is not None:
            self.marginal = DSFMarginal(context_dim=flow_input_dim, **dsf_marginal)

        self.marginal_logdet = None

    def loss(
        self,
        flow_encoded: torch.Tensor,
        mask: torch.BoolTensor,
        true_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss function of the decoder.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            Only the values where the mask is set to False will be considered in the loss function.

        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.
        """
        flow_encoded = _merge_series_time_dims(flow_encoded)

        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch (every sample has the same hist-pred split)
        mask = mask[0, :]  # (series * time steps)

        pred_encoded_flow = flow_encoded[:, ~mask, :]

        pred_true_x = true_value[:, ~mask]

        # Transform to [0,1] using the marginals
        pred_true_u, marginal_logdet = self.marginal.forward_logdet(pred_encoded_flow, pred_true_x)

        self.marginal_logdet = marginal_logdet

        # Loss = negative log likelihood
        return -marginal_logdet
