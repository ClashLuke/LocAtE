import torch

from .activation import nonlinear_function
from .config import (BOTTLENECK, DEVICE, SEPARABLE)
from .inplace_norm import Norm
from .merge import ResModule
from .scale import Scale
from .spectral_norm import SpectralNorm
from .utils import conv_pad_tuple


class RevConvFunction(torch.autograd.Function):
    @staticmethod
    def conv(block_input, conv, padding, groups, *args):
        weight, bias = args
        block_input = torch.nn.functional.batch_norm(block_input,
                                                     running_var=torch.ones(
                                                             block_input.size(
                                                                     1)).to(DEVICE),
                                                     running_mean=torch.zeros(
                                                             block_input.size(
                                                                     1)).to(DEVICE))
        block_input = nonlinear_function(block_input)
        block_input = conv(block_input, weight, bias, padding=padding, groups=groups)
        return block_input

    @staticmethod
    def forward(ctx, block_input, conv, input_tensor_list, append_output, padding,
                split, groups, clear_list,
                *args):
        ctx.conv = conv
        ctx.input_tensor_list = input_tensor_list
        ctx.padding = padding
        ctx.groups = groups
        ctx.clear_list = clear_list
        args0 = args[:split]
        args1 = args[split:]
        ctx.args = [(args0, args1)]
        with torch.no_grad():
            x0, x1 = block_input.chunk(2, 1)
            y0 = RevConvFunction.conv(x1, conv, padding, groups, *args0) + x0
            y1 = RevConvFunction.conv(y0, conv, padding, groups, *args1) + x1
            cat = torch.cat([y0, y1], dim=1)
            if append_output:
                input_tensor_list.append(cat)
            return cat

    @staticmethod
    def backward(ctx, grad_output):
        grad_y1, grad_y0 = grad_output.chunk(2, 1)
        args0, args1 = ctx.args.pop(0)
        conv = ctx.conv
        padding = ctx.padding
        groups = ctx.groups
        with torch.no_grad():
            elem = ctx.input_tensor_list.pop(0)
            if isinstance(elem, torch.Tensor):
                y0, y1 = elem.chunk(2, 1)
            else:  # is tuple
                y0, y1 = elem
            x1 = y1 - RevConvFunction.conv(y0, conv, padding, groups, *args1)
            x0 = y0 - RevConvFunction.conv(x1, conv, padding, groups, *args0)
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
            y0 = RevConvFunction.conv(x1, conv, padding, groups, *args0) + x0
            y1 = RevConvFunction.conv(y0, conv, padding, groups, *args1) + x1
            x0g0, x1g0, *conv0_grad = torch.autograd.grad(y0, (
                    x0, x1, *args0), grad_y1, retain_graph=True)
            x1g1, y0g1, *conv1_grad = torch.autograd.grad(y1, (
                    x1, y0, *args1), grad_y0, retain_graph=True)
            _, _, *conv0_grad = torch.autograd.grad(y0,
                                                    (x0, x1, *args0),
                                                    grad_y1 + y0g1, retain_graph=True)
        if ctx.clear_list:
            ctx.input_tensor_list.clear()
        else:
            with torch.no_grad():
                ctx.input_tensor_list.append((x0.data, x1.data))
        return (torch.cat([y0g1 + x0g0, x1g1 + x1g0], dim=1), *[None] * 7,
                *conv0_grad, *conv1_grad)


class BottleneckRevConvFunction(RevConvFunction):
    @staticmethod
    def conv(block_input, conv, padding, groups, *args: list):
        for weight, bias in zip(args[::2], args[1::2]):
            block_input = torch.nn.functional.batch_norm(block_input,
                                                         running_var=torch.ones(
                                                                 block_input.size(
                                                                         1)).double(),
                                                         running_mean=torch.zeros(
                                                                 block_input.size(
                                                                         1)).double())
            block_input = nonlinear_function(block_input)
            block_input = conv(block_input, weight, bias,
                               padding=padding, groups=groups)
        return block_input


rev_conv_function = RevConvFunction.apply
bottleneck_rev_conv_function = BottleneckRevConvFunction.apply


class RevConv(torch.nn.Module):
    def __init__(self, input_tensor_list, features, kernel_size=1, padding=0,
                 dim=1, groups=1, append_output=True, clear_list=True):
        super().__init__()
        if groups != 1:
            groups //= 2
        features = features // 2
        out_features = features // groups
        conv = getattr(torch.nn.functional, f'conv{dim}d')
        self.weight = torch.nn.Parameter(torch.ones((2, features, out_features,
                                                     *[kernel_size] * dim)))
        self.bias_zeros = torch.zeros(features).to(DEVICE)
        self.padding = padding
        self.conv = conv
        self.input_tensor_list = input_tensor_list
        self.groups = groups
        self.features = features
        self.kernel_size = kernel_size
        self.dim = dim
        self.append_output = append_output
        self.clear_list = clear_list

    def forward(self, function_input):
        weight0, weight1 = self.weight
        return rev_conv_function(function_input, self.conv, self.input_tensor_list,
                                 self.append_output, self.padding, 2, self.groups,
                                 self.clear_list,
                                 weight0, self.bias_zeros, weight1, self.bias_zeros)

    def extra_repr(self):
        return f'features={self.features}, ' \
               f'kernel_size={self.kernel_size}, padding={self.padding}, ' \
               f'groups={self.groups}, dim={self.dim}, ' \
               f'append_output={self.append_output}'


class BottleneckRevConv(torch.nn.Module):
    def __init__(self, input_tensor_list, conv, features, kernel_size=1, padding=0,
                 dim=1, depth=1):
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
        self.conv = conv
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


class ActivatedBaseConv(torch.nn.Module):
    def __init__(self, input_tensor_list, in_features, out_features, transpose, dim,
                 kernel=5, stride=1, pad=2, separable=False, append_output=True,
                 clear_list=True):
        super().__init__()

        conv = getattr(torch.nn, f'Conv{dim}d')

        separable = SEPARABLE or separable
        groups = in_features if separable else 1

        try:
            pad = pad[0]
        except TypeError:
            pass

        mod_2 = in_features % 2 == 0 and out_features % 2 == 0
        second_rev = mod_2 and in_features == out_features
        if mod_2:
            self.scale = Scale(in_features, in_features, stride, transpose)
            self.conv_0 = SpectralNorm(RevConv(input_tensor_list=input_tensor_list,
                                               features=in_features, kernel_size=kernel,
                                               padding=pad, dim=dim, groups=groups,
                                               append_output=append_output and not
                                               second_rev, clear_list=clear_list))
        else:
            self.scale = lambda x: x
            if transpose:
                conv_0_conv = getattr(torch.nn, f'ConvTranspose{dim}d')
            else:
                conv_0_conv = conv
            self.conv_0 = SpectralNorm(conv_0_conv(in_channels=in_features, padding=pad,
                                                   out_channels=in_features,
                                                   kernel_size=kernel, stride=stride,
                                                   groups=groups))

        if second_rev:
            self.conv_1 = SpectralNorm(RevConv(input_tensor_list=input_tensor_list,
                                               features=in_features, dim=dim,
                                               append_output=append_output,
                                               clear_list=False)
                                       )
        else:
            self.conv_1 = conv(
                    kernel_size=1, stride=1, padding=0, out_channels=out_features,
                    bias=False, in_channels=in_features
                    )

    def forward(self, function_input):
        return self.conv_1(nonlinear_function(
                self.conv_0(self.scale(nonlinear_function(function_input)))))


class DeepResidualConv(torch.nn.Module):
    def __init__(self, in_features, out_features, transpose, stride,
                 use_bottleneck=True, dim=2, depth=1, input_tensor_list=None,
                 append_output=True, clear_list=True):
        super().__init__()
        min_features = min(in_features, out_features)
        if use_bottleneck and max(in_features,
                                  out_features) // min_features < BOTTLENECK:
            min_features //= BOTTLENECK
        cnt = [0]
        self.layers = []
        if input_tensor_list is None:
            input_tensor_list = []

        def add_conv(in_features, out_features, residual=True, normalize=False,
                     transpose=False, stride=1, kernel=3, pad=conv_pad_tuple,
                     input_tensor_list=input_tensor_list, append_output=False,
                     clear_list=False):
            layer = ActivatedBaseConv(input_tensor_list, in_features, out_features,
                                      transpose, dim, stride=stride,
                                      kernel=kernel, pad=pad(kernel, stride),
                                      append_output=append_output,
                                      clear_list=clear_list)
            if normalize:
                layer = Norm(in_features, layer, dim)
            if residual and in_features == out_features:
                layer = ResModule(lambda x: x, layer, m=1)
            setattr(self, f'conv_{cnt[0]}', layer)
            cnt[0] += 1
            self.layers.append(layer)

        add_conv(in_features, out_features, False, False,
                 transpose, stride, input_tensor_list=[],
                 append_output=True, clear_list=True)
        i = 0
        for i in range(depth - 2):
            add_conv(out_features, out_features, normalize=bool(i),
                     clear_list=clear_list and i == 0)
        if (depth - 1) > 0:
            add_conv(out_features, out_features, normalize=bool(i),
                     append_output=append_output, clear_list=clear_list and not i)

    def forward(self, function_input: torch.FloatTensor, ):
        for layer in self.layers:
            function_input = layer(function_input)
        return function_input
