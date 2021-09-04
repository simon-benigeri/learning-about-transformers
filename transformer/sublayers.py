"""Sublayers of Encoder and Decoder layers can be found here:
ScaledDotProductAttention,
MultiHeadedAttention,
LayerNorm,
and SublayerConnection"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import clones


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, dropout: nn.Module=None):
        """Compute Scaled Dot Product Attention"""
        super().__init__()
        self.temperature = temperature
        self.dropout = dropout

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None):
        # QK.T/temperature. typically, temperature = sqrt(d_k )
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
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


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout=0.1):
        """input the model size and number of heads"""
        super().__init__()
        assert d_model % n_heads == 0
        # assume d_k = d_v = d_model // n_heads
        self.d_k = d_model // n_heads
        self.heads = n_heads
        self.dropout = dropout
        self.linears = clones(
            module=nn.Linear(in_features=d_model, out_features=self.n_heads * self.d_k, bias=False), N=4)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(self.d_k), dropout=self.dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None):
        # we apply same mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        # query, key, value have shape (n_batches, d_model, d_model)
        query, key, value = [
            # pass them through the pre-attention linear projection layers n_batches x d_model x (n_heads*d_k)
            # then separate the attention heads: n_batches x d_model x n_heads x d_k
            linear(x).view(n_batches, -1, self.heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        # Apply attention on all the projected vectors in batch.
        x, attention_filter = self.attention.forward(query=query, key=key, value=value, mask=mask)
        # concatenate outputs using a view and apply the final linear
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.heads * self.d_k)
        return self.linears[-1](x)