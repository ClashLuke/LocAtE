from .activation import nlinear
from .config import *
from .spectral_norm import SpectralNorm
from .util_modules import *
from .utils import conv_pad_tuple, prod, transpose_pad_tuple


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.num_features = features
        self.bn = nn.BatchNorm2d(features, momentum=0.001, affine=False)
        self.beta = nn.Parameter(torch.zeros(1, features))
        self.gamma = nn.Parameter(torch.ones(1, features))
        torch.nn.init.orthogonal_(self.beta.data)
        torch.nn.init.orthogonal_(self.gamma.data)

    def forward(self, input, factor=None):
        b, c, x, y = input.size()
        if factor is None:
            factor = self.gamma.view(1, c, 1, 1).expand(b, c, x, y)
        else:
            factor = factor.view(b, c, 1, 1).expand(b, c, x, y)
        factor = factor.add(1)
        shift = self.beta.view(1, c, 1, 1).expand(b, c, x, y)
        out = self.bn(input)
        out = out * factor + shift
        return out


class NormModule(nn.Module):
    def __init__(self, features, module):
        super().__init__()
        self.module = module
        self.features = features
        self.bn = ConditionalBatchNorm2d(features)

    def forward(self, input, scale=None):
        return self.bn(self.module(input), scale)

class FeaturePooling(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
    def forward(self, input):
        input_size = list(input.size())
        input_size[1] = self.out_features
        input_view = input.view(*input_size, -1)
        return input_view.mean(dim=-1)


def Scale(in_features, out_features, stride, transpose):
    reslayers = []
    if in_features > out_features:
        reslayers.append(FeaturePooling(out_features))
    elif out_features > in_features:
        reslayers.append(CatModule(lambda x: x, SpectralNorm(nn.Conv2d(in_features, out_features-in_features, 1))))
    if stride > 1:
        if transpose:
            reslayers.append(nn.Upsample(mode='bilinear', scale_factor=stride,
                                         align_corners=False))
        else:
            reslayers.append(nn.AvgPool2d(stride, stride))
    if len(reslayers) > 1:
        return nn.Sequential(*reslayers)
    elif not reslayers:
        return lambda x: x
    else:
        return reslayers[0]


def FactorizedConv(in_features, out_features, kernel_size=1, stride=1, padding=(0, 0),
                   nonlinearity=True, end_nonlinear=True, bias=False, transpose=False,
                   factorize=factorize, scale_first=False, batch_norm=False):
    layers = []
    sample_layers = []
    if transpose:
        conv = nn.ConvTranspose2d
    else:
        conv = nn.Conv2d
    if scale_first:
        function_type = type(lambda x: 0)
        sample_layers.append(Scale(in_features, in_features, stride, transpose))
        if isinstance(sample_layers[0], function_type):
            del sample_layers[0]
        stride = 1
        conv = nn.Conv2d
    if isinstance(kernel_size, int):
        if kernel_size == 1:
            layers.append(conv(in_features, out_features, kernel_size=(1, 1), bias=bias))
        else:
            kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(kernel_size, tuple):
        if factorize:
            min_features = min(in_features, out_features)
            layers.append(conv(in_features, min_features, kernel_size=(kernel_size[0], 1),
                               padding=(padding[0], 0), stride=(stride, 1), bias=bias))
            layers.append(conv(min_features, out_features, kernel_size=(1, kernel_size[1]),
                               padding=(0, padding[1]), stride=(1, stride), bias=bias))
        else:
            layers.append(conv(in_features, out_features, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=bias))
    layers = [SpectralNorm(l) for l in layers]
    if nonlinearity:
        layers.insert(1, nlinear())
    if end_nonlinear:
        layers.append(nlinear())
    if nonlinearity and end_nonlinear and isinstance(layers[-1], nlinear) and isinstance(layers[-2], nlinear):
        del layers[-1]
    if not end_nonlinear and nonlinearity and isinstance(layers[-1], nlinear):
        del layers[-1]
    if batch_norm:
        layers.append(ConditionalBatchNorm2d(out_features))
    layers = sample_layers + layers
    return layers


class FactorizedConvModule(nn.Module):
    def __init__(self, in_features, out_features, kernel, transpose, depth, padding, stride):
        super().__init__()
        min_features = min(in_features, out_features)
        self.final_layer = None
        in_kernel = stride + 2 + int(not transpose)
        cnt = [0]
        self.layers = []
        
        if transpose:
            pad_tuple = transpose_pad_tuple
        else:
            pad_tuple = conv_pad_tuple
        input_pad = pad_tuple(in_kernel, stride)[0]
        default_pad = default_kernel_size // 2
        
        def conv(i, o, k, s, p, c):
            if factorize:
                return [SpectralNorm(c(i, o, (k, 1), (s, 1), (p, 0), bias=False)),
                        SpectralNorm(c(o, o, (1, k), (1, s), (0, p), bias=False))]
            return [SpectralNorm(c(i, o, [k]*2, (s, s), [p]*2, bias=False))]
        
        def conv_layer(i, o, k, s, p, c=nn.Conv2d, cnt=cnt):
            layers = conv(i, o, k, s, p, c)
            for l in layers:
                nlin = nlinear()
                bn = ConditionalBatchNorm2d(i)
                self.layers.append([nlin, bn, l])
                setattr(self, f'nlin_{cnt[0]}', nlin)
                if use_batch_norm:
                    setattr(self, f'bn_{cnt[0]}', bn)
                setattr(self, f'layer_{cnt[0]}', l)
                i = o
                cnt[0] += 1
            
        conv_layer(in_features, min_features, in_kernel, stride, input_pad, c=nn.ConvTranspose2d if transpose else nn.Conv2d)
        if depth == 0:
            return
        for _ in range(2, depth):
            conv_layer(min_features, min_features, default_kernel_size, 1, default_pad)
        conv_layer(min_features, out_features, kernel, 1, padding)

    def forward(self, input, scales=None):
        if scales is None:
            for g in self.layers:
                for l in g:
                    input = l(input)
        else:
            for g, s in zip(self.layers, scales):
                input = l[2](l[1](l[0](input), s))
        return input


class InceptionBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, transpose):
        super().__init__()

        transpose = transpose and stride > 1
        kernels = [factorized_kernel_size] * 3
        scale_first = list(range(3))
        if out_features // len(kernels) * len(kernels) == out_features and out_features // len(kernels) >= min_inception_features:
            out_features = out_features // len(kernels)
        else:
            kernels = [max(kernels)]
            scale_first = [max(scale_first)]

        def params(x, s):
            return dict(kernel=x, stride=stride, padding=conv_pad_tuple(x, stride)[0], transpose=transpose, depth=s)

        c = lambda x, s: FactorizedConvModule(in_features, out_features, **params(x, s))

        layers = [c(k, s) for k, s in zip(kernels, scale_first)]
        self.layers = layers
        for i, L in enumerate(layers):
            setattr(self, f'factorized_conv_{i}', L)

        out_features = out_features * len(kernels)
        self.out = lambda x: torch.cat([l(x) for l in layers], dim=1)
        self.nlin = nlinear()
        self.o_conv = SpectralNorm(nn.Conv2d(out_features, out_features, 1, bias=False))
        self.depth = max(scale_first)

    def forward(self, input, scale=None):
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


class LinearCatModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.module = SpectralNorm(nn.Linear(*args))
        self.nlin = nlinear()

        self.factor = nn.Parameter(torch.ones(1))

    def forward(self, input):
        out = self.module(input)
        return self.nlin(out), out.tanh()


class ResModule(nn.Module):
    def __init__(self, residual_module, layer_module, m=0):
        super(ResModule, self).__init__()
        self.residual_module = residual_module
        self.layer_module = layer_module
        self.gamma = torch.nn.Parameter(torch.zeros(1, 1))
        nn.init.orthogonal_(self.gamma.data).add(m+1)

    def forward(self, input, layer_input=None, scale=None):
        args = [input] if layer_input is None else [layer_input]
        if scale is not None:
            args.append(scale)
        res = self.residual_module(input)
        layer_out = self.layer_module(*args)
        gamma = view_as(self.gamma, layer_out)
        output = res + gamma * layer_out
        return output


class FeatureAttention(nn.Module):
    def __init__(self, in_size, features):
        super().__init__()
        self.conv_0 = SpectralNorm(nn.Conv2d(features, features, kernel_size=(1, in_size), bias=False))
        self.c_nlin = nlinear()
        self.conv_1 = SpectralNorm(nn.Conv2d(features, features, kernel_size=(in_size, 1), bias=False))
        self.f_nlin = nlinear()
        self.conv_o = SpectralNorm(nn.Conv2d(features, features, kernel_size=1, bias=False))
        self.o_nlin = nn.Softmax(dim=1)
        self.expand = Expand(-1, features, in_size, in_size)

    def forward(self, input):
        out = self.expand(self.o_nlin(self.conv_o(self.f_nlin(self.conv_1(self.c_nlin(self.conv_0(input)))))))
        out = out * input
        return out


class SelfAttention(nn.Module):
    def __init__(self, features):
        super(SelfAttention, self).__init__()
        args = [features, features, 1]
        self.conv_0 = SpectralNorm(nn.Conv1d(*args, bias=False))
        self.nlin_0 = nlinear()
        self.conv_1 = SpectralNorm(nn.Conv1d(*args, bias=False))
        self.nlin_1 = nn.Softmax(dim=-1)

    def forward(self, input):
        batch, features, width, height = input.size()
        output = input.view(batch, features, -1)
        output = self.nlin_1(self.conv_1(self.nlin_0(self.conv_0(output))))
        output = output.view(batch, features, width, height)
        output = output * input
        return output


class Block(nn.Module):
    def __init__(self, in_size, in_features, out_features, stride, transpose, block_number, cat_out=True):
        super().__init__()
        self.scale_layer = Scale(in_features, out_features, stride, transpose)
        self.res_module_i = ResModule(lambda x: x, InceptionBlock(in_features, out_features, stride, transpose), m=1)
        if in_size >= min_attention_size and block_number%attention_every_nth_layer == 0:
            self.res_module_f = ResModule(lambda x: x, FeatureAttention(in_size, out_features))
            self.res_module_s = ResModule(lambda x: x, SelfAttention(out_features))
            self.attention = True
        else:
            self.attention = False
        self.cat_out = cat_out 

    def forward(self, input, scales=None):
        scaled = self.scale_layer(input)
        out = self.res_module_i(scaled, input, scales)
        if self.attention:
            out = self.res_module_f(out)
            out = self.res_module_s(out)
        if scales is not None:
            out = out * scales[-1].view(*scales[-1].size(), 1, 1).expand_as(out)
        return out


class BlockBlock(nn.Module):
    def __init__(self, block_count, in_size, features, strides, transpose, mul_channel=False):
        super().__init__()
        self.block_count = block_count

        factors = strides if transpose else [1 / s for s in strides]

        def feature_tuple(idx):
            return features[idx], features[idx + 1]

        def size(idx):
            return int(in_size * prod(factors[:idx + 1]))
    
        def lin_feature_tuple(idx, div):
            if idx%div == 0:
                return feature_tuple(idx//div)
            return [feature_tuple(idx//div)[1]]*2

        blocks = [Block(size(i), *feature_tuple(i), strides[i], transpose, i)
                      for i in range(block_count)]
        self.blocks = blocks

        for i, c in enumerate(blocks):
            setattr(self, f'block_{i}', c)

        depth = blocks[0].res_module_i.layer_module.depth
        #depth += 1  # There is one more multiplication at the end of each block

        if mul_channel:
            mul_blocks = [LinearCatModule(*lin_feature_tuple(i, depth)) for i in range(block_count*depth)]
            self.mul_blocks = mul_blocks

            for i, m in enumerate(mul_blocks):
                setattr(self, f'mul_block_{i}', m)
        self.depth = depth
        self.out_features = feature_tuple(block_count - 1)[1]

    def forward(self, input, mul_channel_input=None):
        for i in range(self.block_count):
            if mul_channel_input is not None:
                operand = []
                for idx in range(self.depth):
                    mul_channel_input, x = self.mul_blocks[i*self.depth+idx](mul_channel_input)
                    operand.append(x)
#                mul_channel_input, operand1 = self.mul_blocks[2*i+1](mul_channel_input)
            else:
                operand = None
            input = self.blocks[i](input, scales=operand)
        return input

