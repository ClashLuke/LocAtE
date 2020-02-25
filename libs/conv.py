import torch

from .activation import nonlinear_function
from .config import (ANTI_ALIAS, BOTTLENECK, FEATURE_MULTIPLIER, SEPARABLE)
from .inplace_norm import Norm
from .merge import ResModule
from .spectral_norm import SpectralNorm
from .utils import conv_pad_tuple, transpose_pad_tuple


class ActivatedBaseConv(torch.nn.Module):
    def __init__(self, in_features, out_features, conv, kernel=3, stride=1, pad=1,
                 dim=2, anti_alias=True):
        super().__init__()
        self.conv_0 = SpectralNorm(conv(in_channels=in_features, kernel_size=kernel,
                                        stride=stride, padding=pad, bias=False,
                                        out_channels=in_features * FEATURE_MULTIPLIER,
                                        groups=in_features if SEPARABLE else 1))
        self.anti_alias = stride > 1 and ANTI_ALIAS and anti_alias
        if self.anti_alias:
            self.aa_layer = getattr(torch.nn, f'AvgPool{dim}d')(kernel_size=3,
                                                                stride=1, padding=1)
        self.conv_1 = SpectralNorm(conv(kernel_size=1, stride=1, padding=0,
                                        out_channels=out_features, bias=False,
                                        in_channels=in_features * FEATURE_MULTIPLIER))

    def forward(self, function_input):
        out = self.conv_0(nonlinear_function(function_input))
        if self.anti_alias:
            out = self.aa_layer(out)
        return self.conv_1(nonlinear_function(out))


class DeepResidualConv(torch.nn.Module):
    def __init__(self, in_features, out_features, transpose, stride,
                 use_bottleneck=True, dim=2, depth=1, anti_alias=True):
        super().__init__()
        min_features = min(in_features, out_features)
        if use_bottleneck and max(in_features,
                                  out_features) // min_features < BOTTLENECK:
            min_features //= BOTTLENECK
        self.final_layer = None
        kernel = (stride + 1) // 2 * 2 + 1 + int(transpose)
        cnt = [0]
        self.layers = []

        if transpose:
            pad_tuple = transpose_pad_tuple
        else:
            pad_tuple = conv_pad_tuple

        default_conv = getattr(torch.nn, f'Conv{dim}d')

        def add_conv(in_features, out_features, residual=True, normalize=False,
                     transpose=False, stride=1, **kwargs):
            conv = getattr(torch.nn,
                           f'ConvTranspose{dim}d') if transpose else default_conv
            layer = ActivatedBaseConv(in_features, out_features, conv, stride=stride,
                                      **kwargs, dim=dim, anti_alias=anti_alias)
            if normalize:
                layer = Norm(in_features, layer, dim)
            if residual and in_features == out_features:
                layer = ResModule(lambda x: x, layer, m=1)
            setattr(self, f'conv_{cnt[0]}', layer)
            cnt[0] += 1
            self.layers.append(layer)

        add_conv(in_features, min_features if depth > 1 else out_features, False, False,
                 transpose, stride, kernel=kernel, pad=pad_tuple(kernel, stride))
        for i in range(depth - 2):
            add_conv(min_features, min_features, default_conv, normalize=bool(i))
        if depth > 1:
            add_conv(min_features, out_features, default_conv,
                     normalize=bool(depth - 2))

    def forward(self, function_input: torch.FloatTensor, ):
        for layer in self.layers:
            function_input = layer(function_input)
        return function_input
