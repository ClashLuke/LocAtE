import torch

from .activation import nonlinear_function
from .config import (BOTTLENECK, FEATURE_MULTIPLIER, SEPARABLE)
from .inplace_norm import Norm
from .merge import ResModule
from .spectral_norm import SpectralNorm
from .utils import conv_pad_tuple, transpose_pad_tuple


class RevConvFunction(torch.autograd.Function):
    @staticmethod
    def conv(block_input, conv, padding, weight, bias):
        block_input = torch.nn.functional.batch_norm(block_input,
                                                     running_var=torch.ones(
                                                             block_input.size(
                                                                     1)).to(DEVICE),
                                                     running_mean=torch.zeros(
                                                             block_input.size(
                                                                     1)).to(DEVICE))
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
        self.bias_zeros = torch.zeros(features).to(DEVICE)
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
class ActivatedBaseConv(torch.nn.Module):
    def __init__(self, in_features, out_features, conv, kernel=5, stride=1, pad=2):
        super().__init__()
        self.conv_0 = SpectralNorm(conv(in_channels=in_features, kernel_size=kernel,
                                        stride=stride, padding=pad, bias=False,
                                        out_channels=in_features * FEATURE_MULTIPLIER,
                                        groups=in_features if SEPARABLE else 1))
        self.conv_1 = SpectralNorm(conv(kernel_size=1, stride=1, padding=0,
                                        out_channels=out_features, bias=False,
                                        in_channels=in_features * FEATURE_MULTIPLIER))

    def forward(self, function_input):
        return self.conv_1(nonlinear_function(
                self.conv_0(nonlinear_function(function_input))))


class DeepResidualConv(torch.nn.Module):
    def __init__(self, in_features, out_features, transpose, stride,
                 use_bottleneck=True, dim=2, depth=1):
        super().__init__()
        min_features = min(in_features, out_features)
        if use_bottleneck and max(in_features,
                                  out_features) // min_features < BOTTLENECK:
            min_features //= BOTTLENECK
        self.final_layer = None
        kernel = stride * 2 + int(not transpose)
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
                                      **kwargs)
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
