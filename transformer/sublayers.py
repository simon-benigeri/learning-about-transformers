"""Sublayers of Encoder and Decoder layers can be found here:
ScaledDotProductAttention, MultiHeadAttention, LayerNorm, SublayerConnection"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class ScaledDotProductAttention(nn.Module):
    def __init__(
            self,
            scaling_factor: float,
            dropout: nn.Module=None
    ):
        """Compute Scaled Dot Product Attention"""
        super().__init__()
        self.scaling = scaling_factor
        self.dropout = dropout

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor=None
    ) -> (Tensor, Tensor):

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
    def __init__(
            self,
            n_heads: int,
            d_model: int,
            # d_k: int,
            # d_v: int,
            dropout=0.1
    ):
        """input the model size and number of heads"""
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model

        # TODO: d_k and d_v as arguments
        #  for now, assume d_k = d_v = d_model // n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.linear_q = nn.Linear(self.d_model, self.n_heads * self.d_k)
        self.linear_k = nn.Linear(self.d_model, self.n_heads * self.d_k)
        self.linear_v = nn.Linear(self.d_model, self.n_heads * self.d_v)
        self.linear_o = nn.Linear(self.n_heads * self.d_v, self.d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model, eps=1e-6)

        self.attention = ScaledDotProductAttention(scaling_factor=math.sqrt(self.d_k), dropout=self.dropout)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor=None
    ) -> (Tensor, Tensor):

        assert query.size(1) == self.d_model
        assert key.size(1) == self.d_model
        assert value.size(1) == self.d_model

        n_batches = query.size(0)
        residual = query
        # we apply same mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)
        # pass query, key, value through the pre-attention linear projection layers and separate  attention heads
        # (n_batches, d_model, d_model) = (n_batches, d_model, (n_heads * d_k)) -> (n_batches, d_model, heads, d_k)
        query = self.linear_q(query).view(n_batches, -1, self.n_heads, self.d_k)
        key = self.linear_k(key).view(n_batches, -1, self.n_heads, self.d_k)
        value = self.linear_v(value).view(n_batches, -1, self.n_heads, self.d_v)

        # transpose for scaled dot product attention:
        # (n_batches, d_model, n_heads, d_k) -> (n_batches, n_heads, d_model, d_k)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        # Apply attention on all the projected vectors in batch.
        x, attention_filter = self.attention.forward(query=query, key=key, value=value, mask=mask)

        # concatenate attention value outputs
        # size: (n_batches, n_heads, d_model, d_k) -> (n_batches, d_model, n_heads * d_k)
        x = x.transpose(1, 2).contiguous().view(n_batches, self.d_model, self.n_heads * self.d_k)

        # residual connection and LayerNorm
        x += residual
        x = self.layer_norm(x)

        # apply final output layer. output shape: (n_batches, d_model, d_k)
        return self.linear_o(x), attention_filter


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


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float=0.1, eps: float=1e-6):
        """implement SublayerConnection, a residual layer followed by a layer norm"""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=size, eps=1e-6)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # x + self.dropout(sublayer(self.norm(x)))
        x = x + sublayer(x)
        return self.dropout(self.layer_norm(x))
