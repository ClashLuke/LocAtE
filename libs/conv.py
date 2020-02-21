import torch

from .activation import NonLinear
from .config import (BOTTLENECK, DEFAULT_KERNEL_SIZE,
                     FACTORIZE, FACTORIZED_KERNEL_SIZE, MIN_INCEPTION_FEATURES,
                     SEPARABLE)
from .spectral_norm import SpectralNorm
from .util_modules import ViewBatch, ViewFeatures
from .utils import conv_pad_tuple, transpose_pad_tuple


class FactorizedConvModule(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel, transpose, depth, padding,
                 stride, use_bottleneck=True, dim=2):
        super().__init__()
        min_features = min(in_features, out_features)
        if use_bottleneck and max(in_features,
                                  out_features) // min_features < BOTTLENECK:
            min_features //= BOTTLENECK
        self.final_layer = None
        in_kernel = stride * 2 + int(not transpose)
        cnt = [0]
        self.layers = []

        if transpose:
            pad_tuple = transpose_pad_tuple
        else:
            pad_tuple = conv_pad_tuple
        input_pad = pad_tuple(in_kernel, stride)[0]
        default_pad = DEFAULT_KERNEL_SIZE // 2

        self.up = ViewFeatures()  # (batch*features, 1)
        self.down = ViewBatch()  # (batch, features)

        def conv(i, o, k, s, p, c):
            if FACTORIZE:
                layer = []
                for d in range(dim):
                    conv_kernel = [1] * dim
                    conv_kernel[d] = k
                    conv_stride = [1] * dim
                    conv_stride[d] = s
                    conv_pad = [0] * dim
                    conv_pad[d] = p
                    layer.append(SpectralNorm(c(i,
                                                o,
                                                conv_kernel, conv_stride, conv_pad,
                                                bias=False,
                                                groups=min(i, o) if SEPARABLE else 1)))
                    i = o
                if SEPARABLE and i % min(i, o) == 0 and o % min(i, o) == 0:
                    layer.append(SpectralNorm(c(o, o, 1, 1, 0, bias=False)))
                return layer
            if SEPARABLE and i % min(i, o) == 0 and o % min(i, o) == 0:
                return [SpectralNorm(c(i, o, [k] * dim, [s] * dim, [p] * dim,
                                       bias=False, groups=min(i, o))),
                        SpectralNorm(c(o, o, [1] * dim, [1] * dim, [0] * dim,
                                       bias=False))]
            return [SpectralNorm(c(i, o, [k] * dim, [s] * dim, [p] * dim, bias=False))]

        default_conv = getattr(torch.nn, f'Conv{dim}d')

        def conv_layer(i, o, k, s, p, c=default_conv, cnt=cnt):
            layers = conv(i, o, k, s, p, c)
            for l in layers:
                nlin = NonLinear()
                self.layers.append([nlin, l])
                setattr(self, f'nlin_{cnt[0]}', nlin)
                setattr(self, f'layer_{cnt[0]}', l)
                cnt[0] += 1

        conv_layer(in_features, min_features, 1, 1, 0)
        conv_layer(min_features, min_features, in_kernel, stride, input_pad,
                   c=getattr(torch.nn,
                             f'ConvTranspose{dim}d') if transpose else default_conv)
        if depth == 0:
            return
        for _ in range(1, depth):
            conv_layer(min_features, min_features, DEFAULT_KERNEL_SIZE, 1, default_pad)
        conv_layer(min_features, out_features, kernel, 1, padding)

    def forward(self, input: torch.FloatTensor, scales=None):
        if SEPARABLE:
            self.down.batch = input.size(0)
        for g in self.layers:
            for l in g:
                input = l(input)
        return input


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, stride, transpose, dim=2):
        super().__init__()

        transpose = transpose and stride > 1
        kernels = [FACTORIZED_KERNEL_SIZE] * 3
        scale_first = list(range(3))
        if out_features // len(kernels) * len(
                kernels) == out_features and out_features // len(
                kernels) >= MIN_INCEPTION_FEATURES:
            out_features = out_features // len(kernels)
        else:
            kernels = [max(kernels)]
            scale_first = [max(scale_first)]

        def params(x, s):
            return dict(kernel=x, stride=stride, padding=conv_pad_tuple(x, stride)[0],
                        transpose=transpose, depth=s)

        def c(x, s):
            return FactorizedConvModule(in_features, out_features, **params(x, s),
                                        dim=dim)

        layers = [c(k, s) for k, s in zip(kernels, scale_first)]
        self.layers = layers
        for i, L in enumerate(layers):
            setattr(self, f'factorized_conv_{i}', L)

        out_features = out_features * len(kernels)
        self.out = lambda *x: torch.cat([l(*x) for l in layers], dim=1)
        self.nlin = NonLinear()
        self.o_conv = SpectralNorm(
                getattr(torch.nn, f'Conv{dim}d')(out_features, out_features, 1,
                                                 bias=False))
        self.depth = max(scale_first)

    def forward(self, function_input):
        out = self.o_conv(self.nlin(self.out(function_input)))
        return out
