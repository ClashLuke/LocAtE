import torch

from .activation import NonLinear
from .config import (BOTTLENECK, DEFAULT_KERNEL_SIZE,
                     FACTORIZED_KERNEL_SIZE, MIN_INCEPTION_FEATURES)
from .spectral_norm import SpectralNorm
from .utils import conv_pad_tuple, transpose_pad_tuple


class RevConvFunction(torch.autograd.Function):
    @staticmethod
    def conv(block_input, conv, padding, weight, bias):
        block_input = torch.nn.functional.batch_norm(block_input,
                                                     running_var=torch.ones(
                                                             block_input.size(
                                                                     1)),
                                                     running_mean=torch.zeros(
                                                             block_input.size(
                                                                     1)))
        block_input = torch.nn.functional.leaky_relu(block_input)
        block_input = conv(block_input, weight, bias, padding=padding)
        return block_input

    @staticmethod
    def forward(ctx, block_input, conv, input_tensor_list, padding, split, *args):
        ctx.conv = conv
        ctx.input_tensor_list = input_tensor_list
        ctx.padding = padding
        args0 = args[:split]
        args1 = args[split:]
        ctx.args = (args0, args1)
        with torch.no_grad():
            x0, x1 = block_input.chunk(2, 1)
            y0 = RevConvFunction.conv(x1, conv, padding, *args0) + x0
            y1 = RevConvFunction.conv(y0, conv, padding, *args1) + x1
            cat = torch.cat([y0, y1], dim=1)
            if not input_tensor_list:
                input_tensor_list.append(cat)
            return cat

    @staticmethod
    def backward(ctx, grad_output):
        grad_y1, grad_y0 = grad_output.chunk(2, 1)
        args0, args1 = ctx.args
        conv = ctx.conv
        padding = ctx.padding
        with torch.no_grad():
            elem = ctx.input_tensor_list.pop(0)
            if isinstance(elem, torch.Tensor):
                x0, x1 = elem.chunk(2, 1)
            else:  # is tuple
                y0, y1 = elem
                x1 = y1 - RevConvFunction.conv(y0, conv, padding, *args1)
                x0 = y0 - RevConvFunction.conv(x1, conv, padding, *args0)
                x1 = x1.data
                x0 = x0.data
        with torch.enable_grad():
            x0.requires_grad_(True)
            x1.requires_grad_(True)
            grad_y0.requires_grad_(True)
            grad_y1.requires_grad_(True)
            x0.retain_grad()
            x1.retain_grad()
            for a in args0:
                a.requires_grad_(True)
                a.retain_grad()
            for a in args1:
                a.requires_grad_(True)
                a.retain_grad()
            y0 = RevConvFunction.conv(x1, conv, padding, *args0) + x0
            y1 = RevConvFunction.conv(y0, conv, padding, *args1) + x1
            x0g0, x1g0, *conv0_grad = torch.autograd.grad(y0, (
                    x0, x1, *args0), grad_y1, retain_graph=True)
            x1g1, y0g1, *conv1_grad = torch.autograd.grad(y1, (
                    x1, y0, *args1), grad_y0, retain_graph=True)
            _, _, *conv0_grad = torch.autograd.grad(y0,
                                                    (x0, x1, *args0),
                                                    grad_y1 + y0g1, retain_graph=True)
        ctx.input_tensor_list.append((x0, x1))
        return (torch.cat([y0g1 + x0g0, x1g1 + x1g0], dim=1), None, None, None, None,
                *conv0_grad, *conv1_grad)


class BottleneckRevConvFunction(RevConvFunction):
    @staticmethod
    def conv(block_input, conv, padding, *args: list):
        for weight, bias in zip(args[::2], args[1::2]):
            block_input = torch.nn.functional.batch_norm(block_input,
                                                         running_var=torch.ones(
                                                                 block_input.size(
                                                                         1)).double(),
                                                         running_mean=torch.zeros(
                                                                 block_input.size(
                                                                         1)).double())
            block_input = torch.nn.functional.leaky_relu(block_input)
            block_input = conv(block_input, weight, bias,
                               padding=padding)
        return block_input


rev_conv_function = RevConvFunction.apply
bottleneck_rev_conv_function = BottleneckRevConvFunction.apply


class RevConv(torch.nn.Module):
    def __init__(self, input_tensor_list, features=0, kernel_size=1,
                 padding=0, dim=1):
        super().__init__()
        features = features // 2
        self.weight = torch.nn.Parameter(torch.ones((2, features, features,
                                                     *[kernel_size] * dim)))
        self.bias_zeros = torch.zeros(features)
        self.padding = padding
        self.conv = getattr(torch, f'conv{dim}d')
        self.input_tensor_list = input_tensor_list

    def forward(self, function_input):
        weight0, weight1 = self.weight
        return rev_conv_function(function_input, self.conv, self.input_tensor_list,
                                 self.padding, 2,
                                 weight0, self.bias_zeros, weight1, self.bias_zeros)


class BottleneckRevConv(torch.nn.Module):
    def __init__(self, input_tensor_list, features, kernel_size=1,
                 padding=0, dim=1, depth=1):
        super().__init__()
        min_features = features // BOTTLENECK

        def mul(x, y):
            return x * y * kernel_size ** dim

        def view(x, y):
            return (x, y, *[kernel_size] * dim)

        self.depth = depth
        self.bottleneck_in = mul(features, min_features)
        self.bottleneck_mid = mul(min_features, min_features) * depth
        self.bottleneck_out = mul(min_features, features)
        self.residual = mul(features, features)
        self.bottleneck_mid += self.bottleneck_in
        self.bottleneck_out += self.bottleneck_mid
        self.residual += self.bottleneck_out
        self.bin_shape = view(features, min_features)
        self.mid_shape = view(min_features, min_features)
        self.out_shape = view(min_features, features)
        self.res_shape = view(features, features)
        self.weight = torch.nn.Parameter(torch.ones(self.residual))
        self.zeros = torch.zeros(features)
        self.mid_zeros = torch.zeros(min_features)
        self.padding = padding
        self.conv = getattr(torch, f'conv{dim}d')
        self.input_tensor_list = input_tensor_list

    def forward(self, function_input):
        bottleneck_in = self.weight[:self.bottleneck_in].view(*self.bin_shape)
        bottleneck_mid = self.weight[self.bottleneck_in:self.bottleneck_mid].view(-1,
                                                                                  *self.mid_shape)
        bottleneck_out = self.weight[self.bottleneck_mid:self.bottleneck_out].view(
                *self.out_shape)
        residual = self.weight[self.bottleneck_out:self.residual].view(*self.res_shape)

        weights_and_biases_1 = [tensor for elem in list(bottleneck_mid) for tensor in
                                [elem, self.zeros]]
        weights_and_biases_1.insert(0, bottleneck_in)
        weights_and_biases_1.insert(1, self.zeros)
        weights_and_biases_1.append(bottleneck_out)
        weights_and_biases_1.append(self.zeros)
        weights_and_biases_1.insert(0, residual)
        weights_and_biases_1.insert(1, self.zeros)
        return bottleneck_rev_conv_function(function_input, self.conv,
                                            self.input_tensor_list, self.padding,
                                            2, weights_and_biases_1)


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
        self.input_tensor_list = []

        if transpose:
            pad_tuple = transpose_pad_tuple
        else:
            pad_tuple = conv_pad_tuple
        input_pad = pad_tuple(in_kernel, stride)[0]
        default_pad = DEFAULT_KERNEL_SIZE // 2

        def conv(f, k, p, c):
            return [SpectralNorm(c(self.input_tensor_list, f, k, p, dim))]

        default_conv = RevConv

        def conv_layer(f, k, p, c=default_conv, cnt=cnt):
            layers = conv(f, k, p, c)
            for l in layers:
                self.layers.append([l])
                setattr(self, f'layer_{cnt[0]}', l)
                cnt[0] += 1

        initial_layer = getattr(torch.nn,
                                f'ConvTranspose{dim}d' if transpose else
                                f'Conv{dim}d')(
                in_channels=in_features, out_channels=out_features,
                kernel_size=in_kernel,
                stride=stride,
                padding=input_pad
                )
        self.layers.append([initial_layer])
        for _ in range(depth):
            conv_layer(out_features, kernel, padding)

    def forward(self, input: torch.FloatTensor, scales=None):
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
