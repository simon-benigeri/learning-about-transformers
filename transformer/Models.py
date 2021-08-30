import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    """Implement positional encoding fuction"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Registers a buffer that should not to be considered a model parameter
        # to the module's 'state_dict'
        self.register_buffer('positional_encodings', self._sinusoidal_positional_encoding(max_len, d_model))

    def _sinusoidal_positional_encoding(self, max_len, d_model):
        """Compute the sinusoid positional encodings"""
        encodings = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1)
        # e^([2i/d] * - log(10000) == (1/ (10000 ^ [2i/d])
        division_term = torch.pow(10000.0, -1 * torch.arange(0, d_model, 2) / d_model)
        # division_term = torch.exp(torch.arange(0, d_model, 2) / d_model * -math.log(10000.0))
        encodings[:, 0::2] = torch.sin(positions * division_term)
        encodings[:, 1::2] = torch.cos(positions * division_term)
        return encodings.unsqueeze(0)

    def forward(self, x):
        x = x + Variable(self.positional_encodings[:, x.size(1)], requires_grad=True)
        return self.dropout(x)

