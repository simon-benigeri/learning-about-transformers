"""Sublayers of Encoder and Decoder layers can be found here:
ScaledDotProductAttention, MultiHeadAttention, LayerNorm, SublayerConnection"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer.utils import clones


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scaling_factor: float, dropout: nn.Module=None):
        """Compute Scaled Dot Product Attention"""
        super().__init__()
        self.scaling = scaling_factor
        self.dropout = dropout

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> (Tensor, Tensor):
        # QK.T/temperature. typically, temperature = sqrt(d_k )
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling
        # apply optional mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax scores
        attention_filter = F.softmax(scores, dim=-1)
        # apply optional dropout
        if self.dropout is not None:
            attention_filter = self.dropout(attention_filter)
        # matrix multiply attention filter with values
        return torch.matmul(attention_filter, value), attention_filter


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout=0.1):
        """input the model size and number of heads"""
        super().__init__()
        assert d_model % n_heads == 0
        # assume d_k = d_v = d_model // n_heads
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = dropout
        self.linears = clones(module=nn.Linear(in_features=d_model,
                                               out_features=self.n_heads * self.d_k,
                                               bias=False),
                              N=4)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = ScaledDotProductAttention(scaling_factor=math.sqrt(self.d_k), dropout=self.dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
        # we apply same mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        # 1. pass query, key, value through the pre-attention linear projection layers
        # 2. separate the attention heads
        # shape of query, key, value:
        # (n_batches, d_model, d_model) -> (n_batches, d_model, heads, d_k) -> (n_batches, heads, d_model, d_k)
        query, key, value = [
            linear(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        # Apply attention on all the projected vectors in batch.
        filtered_value, attention_filter = self.attention.forward(query=query, key=key, value=value, mask=mask)
        # concatenate attention value outputs
        # size: (n_batches, heads, d_model, d_k) -> (n_batches, d_model, n_heads * d_k)
        filtered_value = filtered_value.transpose(1, 2).contiguous().view(n_batches, -1, self.n_heads * self.d_k)
        # apply final output layer. output shape: (n_batches, d_model, d_k)
        return self.linears[-1](filtered_value)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int=512, d_ff: int=2048, dropout: float=0.1):
        """implement Position-wise FFN(x) = max(0, xW_1 + b_1)W_2 + b_2"""
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        return self.linear_2(x)


class LayerNorm(nn.Module):
    def __init__(self):
        """implement LayerNorm"""
        super().__init__()

    def forward(self):

        return


class SublayerConnection(nn.Module):
    def __init__(self):
        """implement SublayerConnection"""
        super().__init__()

    def forward(self):

        return
