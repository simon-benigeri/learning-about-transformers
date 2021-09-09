import math
import torch
import torch.nn as nn
from torch import Tensor


class Embeddings(nn.Module):
    """Implement word embedding module"""
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement positional encoding function"""
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # PEs are not model parameters -> register them as a buffer to module's 'state_dict'
        self.register_buffer('positional_encodings', self._sinusoidal_positional_encoding(max_len, d_model))

    def _sinusoidal_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        """Compute the sinusoid positional encodings"""
        encodings = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1)

        division_term = torch.pow(10000.0, -1 * torch.arange(0, d_model, 2) / d_model)
        encodings[:, 0::2] = torch.sin(positions * division_term)
        encodings[:, 1::2] = torch.cos(positions * division_term)

        return encodings.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.positional_encodings[:, :x.size(1)].detach().clone()
        return self.dropout(x)

