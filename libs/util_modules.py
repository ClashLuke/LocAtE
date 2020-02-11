import torch
from torch import nn

from .utils import view_expand


class Expand(nn.Module):
    def __init__(self, *target_size):
        super(Expand, self).__init__()
        self.target_size = target_size

    def forward(self, input):
        return view_expand(self.target_size, input)


class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, *input):
        return torch.cat(input, self.dim)


class Sum(nn.Module):
    def __init__(self, target_features):
        super(Sum, self).__init__()
        self.target_features = target_features
        self.norm = norm(target_features)
        self.nlin = nlinear()

    def forward(self, input):
        return self.nlin(self.norm(torch.stack(input.split(self.target_features, dim=1), dim=0).sum(dim=0)))
