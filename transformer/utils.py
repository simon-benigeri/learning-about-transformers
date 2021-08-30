import copy
import torch.nn as nn

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])