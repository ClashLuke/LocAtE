from .activation import nlinear
from .config import *
from .spectral_norm import SpectralNorm
from .util_modules import *
from .utils import conv_pad_tuple, prod, transpose_pad_tuple


class FeaturePooling(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, input):
        input_size = list(input.size())
        input_size[1] = self.out_features
        input_view = input.view(*input_size, -1)
        return input_view.mean(dim=-1)


def Scale(in_features, out_features, stride, transpose, dim=2):
    reslayers = []
    if in_features > out_features:
        reslayers.append(FeaturePooling(out_features))
    elif out_features > in_features:
        reslayers.append(CatModule(lambda x: x, SpectralNorm(
                getattr(nn, f'Conv{dim}d')(in_features, out_features - in_features,
                                           1))))
    if stride > 1:
        if transpose:
            reslayers.append(nn.Upsample(mode='bilinear', scale_factor=stride,
                                         align_corners=False))
        else:
            reslayers.append(getattr(nn, f'AvgPool{dim}d')(stride, stride))
    if len(reslayers) > 1:
        return nn.Sequential(*reslayers)
    elif not reslayers:
        return lambda x: x
    else:
        return reslayers[0]


class ViewBatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch = 1

    def forward(self, input):
        _, _, *size = input.size()
        return input.view(self.batch, -1, *size)


class ViewFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = 1

    def forward(self, input):
        _, _, *size = input.size()
        return input.view(-1, self.features, *size)


class FactorizedConvModule(nn.Module):
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
        default_pad = default_kernel_size // 2

        self.up = ViewFeatures()  # (batch*features, 1)
        self.down = ViewBatch()  # (batch, features)

        def conv(i, o, k, s, p, c):
            if factorize:
                layer = []
                if separable:
                    layer.append(self.up)
                for d in range(dim):
                    conv_kernel = [1] * dim
                    conv_kernel[d] = k
                    conv_stride = [1] * dim
                    conv_stride[d] = s
                    conv_pad = [0] * dim
                    conv_pad[d] = p
                    layer.append(SpectralNorm(c(1 if separable else i,
                                                1 if separable else o,
                                                conv_kernel, conv_stride, conv_pad,
                                                bias=False)))
                    if not separable:
                        i = o
                if separable:
                    layer.append(self.down)
                    layer.append(SpectralNorm(c(i, o, 1, 1, 0, bias=False)))
                return layer
            elif separable:
                return [self.up,
                        SpectralNorm(
                                c(1, 1, [k] * dim, [s] * dim, [p] * dim, bias=False)),
                        self.down,
                        SpectralNorm(
                                c(i, o, [1] * dim, [1] * dim, [0] * dim, bias=False))]
            return [SpectralNorm(c(i, o, [k] * dim, [s] * dim, [p] * dim, bias=False))]

        default_conv = getattr(nn, f'Conv{dim}d')

        def conv_layer(i, o, k, s, p, c=default_conv, cnt=cnt):
            layers = conv(i, o, k, s, p, c)
            for l in layers:
                nlin = nlinear()
                self.layers.append([nlin, l])
                setattr(self, f'nlin_{cnt[0]}', nlin)
                setattr(self, f'layer_{cnt[0]}', l)
                cnt[0] += 1

        conv_layer(in_features, min_features, 1, 1, 0)
        conv_layer(min_features, min_features, in_kernel, stride, input_pad,
                   c=getattr(nn, f'ConvTranspose{dim}d') if transpose else default_conv)
        if depth == 0:
            return
        for _ in range(1, depth):
            conv_layer(min_features, min_features, default_kernel_size, 1, default_pad)
        conv_layer(min_features, out_features, kernel, 1, padding)

    def forward(self, input: torch.FloatTensor, scales=None):
        if separable:
            self.down.batch = input.size(0)
        for g in self.layers:
            for l in g:
                input = l(input)
        return input


class InceptionBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, transpose, dim=2):
        super().__init__()

        transpose = transpose and stride > 1
        kernels = [factorized_kernel_size] * 3
        scale_first = list(range(3))
        if out_features // len(kernels) * len(
                kernels) == out_features and out_features // len(
                kernels) >= min_inception_features:
            out_features = out_features // len(kernels)
        else:
            kernels = [max(kernels)]
            scale_first = [max(scale_first)]

        def params(x, s):
            return dict(kernel=x, stride=stride, padding=conv_pad_tuple(x, stride)[0],
                        transpose=transpose, depth=s)

        c = lambda x, s: FactorizedConvModule(in_features, out_features, **params(x, s),
                                              dim=dim)

        layers = [c(k, s) for k, s in zip(kernels, scale_first)]
        self.layers = layers
        for i, L in enumerate(layers):
            setattr(self, f'factorized_conv_{i}', L)

        out_features = out_features * len(kernels)
        self.out = lambda *x: torch.cat([l(*x) for l in layers], dim=1)
        self.nlin = nlinear()
        self.o_conv = SpectralNorm(
                getattr(nn, f'Conv{dim}d')(out_features, out_features, 1, bias=False))
        self.depth = max(scale_first)

    def forward(self, input):
        out = self.o_conv(self.nlin(self.out(input)))
        return out


# In[9]:

def view_as(tensor, x):
    return tensor.view(1, -1, 1, 1).expand_as(x)


class CatModule(nn.Module):
    def __init__(self, residual_module, layer_module):
        super().__init__()
        self.residual_module = residual_module
        self.layer_module = layer_module

    def forward(self, input, layer_input=None, scale=None):
        args = [input] if layer_input is None else [layer_input]
        if scale is not None:
            args.append(scale)
        res, layer_out = self.residual_module(input), self.layer_module(*args)
        output = torch.cat([res, layer_out], dim=1)
        return output


class LinearModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = SpectralNorm(nn.Linear(*args))
        self.nlin = nlinear()

    def forward(self, input):
        out = self.module(input)
        return self.nlin(out), out


class Norm(nn.Module):
    def __init__(self, features, module, dim=2):
        super().__init__()
        self.i_norm = getattr(nn, f'BatchNorm{dim}d')(features, affine=False)
        self.module = module

    def forward(self, input, scale=None):
        module_input = self.i_norm(input)
        if scale is not None:
            module_input = module_input.mul(scale)
        return self.module(module_input)


class ResModule(nn.Module):
    def __init__(self, residual_module, layer_module, m=0):
        super(ResModule, self).__init__()
        self.residual_module = residual_module
        self.layer_module = layer_module
        self.merge = ResidualFunction.apply
        self.gamma = nn.Parameter(torch.ones((1, 1)))
        torch.nn.init.orthogonal_(self.gamma.data)
        self.gamma.data.add_(m + 1)

    def forward(self, input, layer_input=None, scale=None):
        args = [input] if layer_input is None else [layer_input]
        if scale is not None:
            args.append(scale)
        res = self.residual_module(input)
        layer_out = self.layer_module(*args)
        gamma = self.gamma.view(*[1] * len(layer_out.size())).expand_as(layer_out)
        return self.merge(res, layer_out, gamma)


class ResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, attention, gamma):
        ctx.save_for_backward(input, attention, gamma)
        # x: input, y: attention, z: gamma
        # (z*y+1)*x
        gamma_attention = attention.mul(gamma)
        gamma_attention.add_(1)
        gamma_attention.mul_(input)
        return gamma_attention

    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        dz = x.mul(grad_output)
        dy = dz.mul(z)
        dx = y.mul(z)
        dx.add_(1)
        dx.mul_(grad_output)
        dz.mul_(x)
        return dx, dy, dz


def FeatureAttention(in_size, features, dim=2):
    bfeatures = features // BOTTLENECK
    layers = []
    default_conv = getattr(nn, f'Conv{dim}d')
    input_features = features
    if separable:
        in_view = ViewFeatures()
        layers.append(in_view)
    for i in range(dim):
        kernel_size = [1] * dim
        kernel_size[i] = in_size
        layers.extend([SpectralNorm(default_conv(1 if separable else input_features,
                                                 1 if separable else bfeatures,
                                                 kernel_size=kernel_size,
                                                 bias=False)),
                       nlinear()])
        input_features = bfeatures
    if separable:
        out_view = ViewFeatures()
        out_view.features = features
        layers.append(out_view)
        bfeatures = features
    layers.extend(
            [SpectralNorm(default_conv(bfeatures, features, kernel_size=1, bias=False)),
             nn.Softmax(dim=1),
             Expand(-1, features, *([in_size] * dim))])
    return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    def __init__(self, features):
        super(SelfAttention, self).__init__()
        args = [features, features, 1]
        self.conv_0 = SpectralNorm(nn.Conv1d(*args, bias=False))
        self.nlin_0 = nlinear()
        self.conv_1 = SpectralNorm(nn.Conv1d(*args, bias=False))
        self.nlin_1 = nn.Softmax(dim=-1)

    def forward(self, input):
        batch, features, *size = input.size()
        output = input.view(batch, features, -1)
        output = self.nlin_1(self.conv_1(self.nlin_0(self.conv_0(output))))
        output = output.view(batch, features, *size)
        return output


class TanhMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        t = x.tanh()
        t.add_(1)
        t.mul_(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        dy = x.tanh()
        dy.add_(1)
        dx = x.cosh()
        dx.pow_(2)
        dx.reciprocal_()
        dx.mul_(y)
        dx.mul_(grad_output)
        dy.mul_(grad_output)
        return dx, dy


class Block(nn.Module):
    def __init__(self, in_size, in_features, out_features, stride, transpose,
                 block_number, cat_out=True, dim=2):
        super().__init__()
        self.scale_layer = Scale(in_features, out_features, stride, transpose, dim=dim)
        self.res_module_i = ResModule(lambda x: x,
                                      Norm(in_features,
                                           InceptionBlock(in_features, out_features,
                                                          stride, transpose, dim=dim),
                                           dim=dim),
                                      m=1)
        if in_size >= min_attention_size and block_number % attention_every_nth_layer == 0:
            self.res_module_f = ResModule(lambda x: x, Norm(out_features,
                                                            FeatureAttention(in_size,
                                                                             out_features,
                                                                             dim=dim),
                                                            dim=dim))
            self.res_module_s = ResModule(lambda x: x, Norm(out_features,
                                                            SelfAttention(out_features),
                                                            dim=dim))
            self.attention = True
        else:
            self.attention = False
        self.cat_out = cat_out

    def forward(self, input, scales=None):
        if scales is None:
            scales = [None] * 4
        scaled = self.scale_layer(input)
        out = self.res_module_i(scaled, input, scales[0])
        if self.attention:
            out = self.res_module_f(out, scale=scales[1])
            out = self.res_module_s(out, scale=scales[2])
        if scales[-1] is not None:
            out = TanhMul.apply(scales[-1].expand_as(out), out)
        return out


class BlockBlock(nn.Module):
    def __init__(self, block_count, in_size, features, strides, transpose,
                 mul_channel=False, dim=2):
        super().__init__()
        self.block_count = block_count

        factors = strides if transpose else [1 / s for s in strides]

        def feature_tuple(idx):
            return features[idx], features[idx + 1]

        def size(idx):
            return int(in_size * prod(factors[:idx + 1]))

        def lin_feature_tuple(idx, div):
            if idx % div == 0:
                return feature_tuple(idx // div)
            return [feature_tuple(idx // div)[1]] * 2

        blocks = [Block(size(i), *feature_tuple(i), strides[i], transpose, i, dim=dim)
                  for i in range(block_count)]
        self.blocks = blocks

        for i, c in enumerate(blocks):
            setattr(self, f'block_{i}', c)
        sums = [0]
        depths = []
        if mul_channel:
            mul_blocks = []
            prev_out = 0
            for i in range(block_count):
                scales = 2 * blocks[i].attention
                depths.append(2 + scales)
                sums.append(sums[-1] + scales + 2)
                inp, out = feature_tuple(i)
                if prev_out and prev_out != inp:
                    group_inp = prev_out
                else:
                    group_inp = inp
                mul_blocks.append(
                        LinearModule(group_inp + input_vector_z * bool(i), inp))
                mul_blocks.append(LinearModule(inp + input_vector_z, out))
                attention_scales = [LinearModule(out + input_vector_z, out) for _ in
                                    range(scales)]
                mul_blocks.extend(attention_scales)

            self.mul_blocks = mul_blocks

            for i, m in enumerate(mul_blocks):
                setattr(self, f'mul_block_{i}', m)

        self.depths = depths
        self.sums = sums
        self.out_features = feature_tuple(block_count - 1)[1]

    def forward(self, input, z=None):
        x = None
        for i in range(self.block_count):
            if z is not None:
                operand = []
                for idx in range(self.depths[i]):
                    x, f = self.mul_blocks[self.sums[i] + idx](
                            z if x is None else torch.cat([z, x], dim=1))
                    operand.append(f.view(*f.size(), 1, 1))
            #                mul_channel_input, operand1 = self.mul_blocks[2*i+1](mul_channel_input)
            else:
                operand = None
            input = self.blocks[i](input, scales=operand)
        return input
