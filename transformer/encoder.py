"""Encoder and EncoderLayer"""
import torch.nn as nn
from torch import Tensor

from utils import clones
from sublayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
	def __init__(
			self,
			n_heads: int,
			d_model: int,
			# d_k: int,
			# d_v: int,
			d_ff: int,
			dropout: float=0.1):
		super().__init__()
		self.self_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
		self.feedforward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
		self.d_model = d_model

	def forward(self, x, self_attention_mask):
		x, self_attention_filter = self.self_attention(query=x, key=x, value=x, mask=self_attention_mask)
		x = self.feedforward(x)
		return x, self_attention_filter


class Encoder(nn.Module):
	def __init__(
			self,
			N: int,
			layer: EncoderLayer
	):
		super().__init__()
		self.layers = clones(layer, N)
		self.layer_norm = nn.LayerNorm(normalized_shape=layer.d_model, eps=1e-6)

	def forward(self):
		pass

