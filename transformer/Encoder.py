import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from utils import *

class Encoder(nn.Module):
	def __init__(self, layer, N):
		super().__init__()
		self.layers = clones(layer, N)

