import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


def clones(module, N):
	return nn.ModuleList([deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self

