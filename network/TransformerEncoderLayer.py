# Standard Transformer Encoder Architecture code written by ChatGPT 4.0
# Used to make slight modifications for masking the effect of specific time series by making their attention score zero


import torch.nn as nn
from network import MultiheadAttention as MLA


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, masked_time_series, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.masked_time_series = masked_time_series
        self.self_attn = MLA.MultiheadAttention(d_model, nhead, self.masked_time_series, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
