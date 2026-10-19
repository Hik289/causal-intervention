import torch
from torch import nn

class InterventionRanker(nn.Module):
    pass
    def __init__(self, dz, ddelta, hidden=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dz+ddelta,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self, z, h_delta):
        return self.net(torch.cat([z,h_delta],-1)).squeeze(-1)
