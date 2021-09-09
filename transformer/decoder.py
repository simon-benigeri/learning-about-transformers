"""Decoder and DecoderLayer"""
import torch
import torch.nn as nn
from torch import Tensor

from utils import clones
from sublayers import MultiHeadAttention, PositionwiseFeedForward


class DecoderLayer(nn.Module):
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
		self.encoder_self_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
		self.feedforward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
		self.d_model = d_model

	def forward(self, x, self_attention_mask, decoder_encoder_self_attention_mask):
		x, self_attention_filter = self.self_attention(query=x, key=x, value=x, mask=self_attention_mask)
		x, encoder_self_attention_filter = self.encoder_self_attention(query=x, key=x, value=x, mask=decoder_encoder_self_attention_mask)
		x = self.feedforward(x)
		return x, self_attention_filter, encoder_self_attention_filter


class Decoder(nn.Module):
	def __init__(
			self,
			N: int,
			layer: DecoderLayer
	):
		super().__init__()
		self.layers = clones(layer, N)
		self.layer_norm = nn.LayerNorm(normalized_shape=layer.d_model, eps=1e-6)

	def forward(self):
		pass