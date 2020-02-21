import torch
from torch import nn

from .attention import SelfAttention, feature_attention
from .config import (ATTENTION_EVERY_NTH_LAYER, INPUT_VECTOR_Z,
                     MIN_ATTENTION_SIZE)
from .conv import InceptionBlock
from .inplace_norm import Norm
from .linear import LinearModule
from .merge import ResModule
from .scale import Scale
from .utils import prod


class TanhMul(torch.autograd.Function):
    @staticmethod
    # skipcq
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        t = x.tanh()
        t.add_(1)
        t.mul_(y)
        return t

    @staticmethod
    # skipcq
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors  # skipcq
        dy = x.tanh()  # skipcq
        dy.add_(1)
        dx = x.cosh()  # skipcq
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
        if (in_size >= MIN_ATTENTION_SIZE and
                block_number % ATTENTION_EVERY_NTH_LAYER == 0):
            self.res_module_f = ResModule(lambda x: x,
                                          Norm(out_features,
                                               feature_attention(in_size,
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

    def forward(self, function_input, scales=None):
        if scales is None:
            scales = [None] * 4
        scaled = self.scale_layer(function_input)
        out = self.res_module_i(scaled, function_input, scales[0])
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
            out = in_size * prod(factors[:idx + 1])
            nout = int(out + 1 - 1e-12)
            print(out, nout)
            return nout

        blocks = [Block(size(i), *feature_tuple(i), strides[i], transpose, i, dim=dim)
                  for i in range(block_count)]
        self.blocks = blocks

        for i, block in enumerate(blocks):
            setattr(self, f'block_{i}', block)
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
                        LinearModule(group_inp + INPUT_VECTOR_Z * bool(i), inp))
                mul_blocks.append(LinearModule(inp + INPUT_VECTOR_Z, out))
                attention_scales = [LinearModule(out + INPUT_VECTOR_Z, out) for _ in
                                    range(scales)]
                mul_blocks.extend(attention_scales)

            self.mul_blocks = mul_blocks

            for i, block in enumerate(mul_blocks):
                setattr(self, f'mul_block_{i}', block)

        self.depths = depths
        self.sums = sums
        self.out_features = feature_tuple(block_count - 1)[1]

    def forward(self, function_input, noise=None):
        next_input = None
        for i in range(self.block_count):
            if noise is None:
                operand = None
            else:
                operand = []
                for idx in range(self.depths[i]):
                    if next_input is None:
                        next_input = noise
                    else:
                        next_input = torch.cat([noise, next_input], dim=1)
                    next_input, factor = self.mul_blocks[self.sums[i] + idx](next_input)
                    operand.append(factor.view(*factor.size(), 1, 1))
            function_input = self.blocks[i](function_input, scales=operand)
        return function_input
