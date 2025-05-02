# Standard MultiHeadAttention mechanism code written by ChatGPT 4.0
# Used to make slight modifications for masking the effect of specific time series by making their attention score zero


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, masked_time_series, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.masked_time_series = masked_time_series

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # Compute Q, K, V in one go
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def shape(x):
            return x.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)  # [B, H, T, D]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        if self.masked_time_series is not None:
            print("mask is called with the number:", self.masked_time_series)
            assert self.masked_time_series < tgt_len
            attn_scores[:, :, :, self.masked_time_series] = -torch.inf

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim).transpose(0, 1)

        return self.out_proj(output), attn_weights
