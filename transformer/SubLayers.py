import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import clones

def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 mask: Tensor=None,
                                 dropout: nn.Dropout=None
                                 ) -> (Tensor, Tensor):
    """Compute Scaled Dot Product Attention. Attention(Q,K,V) = softmax(QK.T/sqrt(d_k)) V"""
    d_k = key.size(-1)
    # QK.T/sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # apply optional mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # softmax scores
    attention_filter = F.softmax(scores, dim=-1)
    # apply optional dropout
    if dropout is not None:
        attention_filter = dropout(attention_filter)
    # matmul with values
    return torch.matmul(attention_filter, value), attention_filter

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout=0.1):
        """input the model size and number of heads"""
        super().__init__()
        assert d_model % n_heads == 0
        # assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.heads = n_heads
        self.linear_layers = clones(module=nn.Linear(in_features=d_model, out_features=d_model), N=4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None):
        if mask is not None:
            # we apply same mask to all heads
            mask = mask.unsqueeze(1)
        num_batches = query.size(0)

    pass