import torch

from .activation import NonLinear
from .config import BOTTLENECK, SEPARABLE
from .spectral_norm import SpectralNorm
from .util_modules import Expand


def feature_attention(in_size, features, dim=2):
    bfeatures = features // BOTTLENECK
    layers = []
    default_conv = getattr(torch.nn, f'Conv{dim}d')
    input_features = features
    min_features = min(input_features, bfeatures)
    if (SEPARABLE and
            input_features % min_features == 0 and
            bfeatures % min_features == 0):
        layers.append(SpectralNorm(default_conv(input_features, bfeatures,
                                                kernel_size=[in_size] * dim,
                                                bias=False,
                                                groups=min_features)))
    else:
        for i in range(dim):
            kernel_size = [1] * dim
            kernel_size[i] = in_size
            layers.extend([SpectralNorm(default_conv(input_features,
                                                     bfeatures,
                                                     kernel_size=kernel_size,
                                                     bias=False,
                                                     groups=1)),
                           NonLinear()])
            input_features = bfeatures
    layers.extend([SpectralNorm(default_conv(bfeatures, features,
                                             kernel_size=1, bias=False)),
                   torch.nn.Softmax(dim=1),
                   Expand(-1, features, *([in_size] * dim))])
    return torch.nn.Sequential(*layers)


class SelfAttention(torch.nn.Module):
    def __init__(self, features):
        super(SelfAttention, self).__init__()
        args = [features, features, 1]
        self.conv_0 = SpectralNorm(torch.nn.Conv1d(*args, bias=False))
        self.nlin_0 = NonLinear()
        self.conv_1 = SpectralNorm(torch.nn.Conv1d(*args, bias=False))
        self.nlin_1 = torch.nn.Softmax(dim=-1)

    def forward(self, function_input):
        batch, features, *size = function_input.size()
        output = function_input.view(batch, features, -1)
        output = self.nlin_1(self.conv_1(self.nlin_0(self.conv_0(output))))
        output = output.view(batch, features, *size)
        return output
