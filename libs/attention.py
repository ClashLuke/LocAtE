import torch

from .activation import NonLinear
from .config import BOTTLENECK, SEPARABLE
from .spectral_norm import SpectralNorm
from .util_modules import Expand, ViewFeatures


def feature_attention(in_size, features, dim=2):
    bfeatures = features // BOTTLENECK
    layers = []
    default_conv = getattr(torch.nn, f'Conv{dim}d')
    input_features = features
    if SEPARABLE:
        in_view = ViewFeatures()
        layers.append(in_view)
    for i in range(dim):
        kernel_size = [1] * dim
        kernel_size[i] = in_size
        layers.extend([SpectralNorm(default_conv(1 if SEPARABLE else input_features,
                                                 1 if SEPARABLE else bfeatures,
                                                 kernel_size=kernel_size,
                                                 bias=False)),
                       NonLinear()])
        input_features = bfeatures
    if SEPARABLE:
        out_view = ViewFeatures()
        out_view.features = features
        layers.append(out_view)
        bfeatures = features
    layers.extend(
            [SpectralNorm(default_conv(bfeatures, features, kernel_size=1, bias=False)),
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
