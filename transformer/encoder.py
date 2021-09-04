"""Encoder and EncoderLayer"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones
from sublayers import MultiHeadAttention


class Encoder(nn.Module):
	def __init__(self, layer, N):
		super().__init__()
		self.layers = clones(layer, N)

