import torch

from .activation import nonlinear_function
from .spectral_norm import SpectralNorm


class LinearModule(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = SpectralNorm(torch.nn.Linear(*args))

    def forward(self, function_input):
        out = self.module(function_input)
        return nonlinear_function(out), out
