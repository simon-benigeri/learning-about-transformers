import math, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import *

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float=None, attention_dropout: float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=attention_dropout)
        # self.temperature = temperature

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
        d_k = key.size(-1)
        # matmul Q, K.T and scale by dividing by temperature (usually d_k ** 0.5)
        # scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_filter = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attention_filter, value)
        return attention_filter, output


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout=0.1):
        """input the model size and number of heads"""
        super().__init__()
        assert d_model % heads == 0
        # assume d_v always equals d_k
        self.d_k = d_model // heads
        self.heads = heads
        # TODO: finish this
        # self.linear_layers = clones(nn.Linear(in_features=))
    pass